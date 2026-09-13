use crate::geometry::backend::GeometryKind;
use crate::geometry::neighbor_lists::NeighborLists;
use crate::geometry::Geometry;
use crate::mpi_context::MpiContext;
use crate::track::{HistorySelection, NoOpTracker, RealTracker, TrackStorage, Tracker};
use crate::util::fast_rng::FastRng;
use crate::variance_reduction::{SurvivalBiasing, VarianceReduction};
use rand::RngExt;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{IsTerminal, Write};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::sync::Mutex;
use yamc_physics::gpu::flat::inelastic_flat::InelasticFlatCache;
use yamc_physics::util::bank::ParticleBank;
use yamc_source::source::{ParticleSource, SourceSelector};
use yamc_tallies::tally::Tally;
use yani_transmute::TransmutationTallies;

// Re-export the model-configuration enums (moved to `crate::options`) so that
// `yamc::model::Verbose` / `yamc::model::TrackingMode` keep resolving for
// external crates (e.g. yamc-python).
pub use crate::options::{TrackingMode, Verbose};

// Transport / collision free functions (moved to `crate::transport`).
// `run_internal` calls into these (tests import `handle_photon_collision`
// directly inside the `tests` module).
use crate::transport::{
    cell_mean_chord, transport_particle, transport_particle_woodcock, TransportCtx,
};

// Debug-instrumentation cluster (statics + dual-cfg helpers) moved to
// `crate::transport::debug`. The only call sites left in `model.rs` are the
// batch-summary counters, read inside a `#[cfg(feature = "debug_runtime")]`
// block; gate the import to match so the default build stays warning-free.
#[cfg(feature = "debug_runtime")]
use crate::transport::debug::{get_debug_counters, get_elastic_scatter_count};

/// Per-batch and per-event diagnostic counters for photon transport debugging.
/// Implementation lives in `yamc-physics` so the photon-secondary samplers
/// (bremsstrahlung, D1S) can write into the same counters that this engine
/// reads at batch boundaries. Compiled away when `debug_diagnostics` is off.
#[cfg(feature = "debug_diagnostics")]
pub use yamc_physics::photon_diag;

/// Mutex serialising photon / TTB global-state setup.
///
/// The global TTB energy grids go through a LINEAR → LOG conversion during
/// `model.simulate_transport()`. When cargo test runs multiple model tests in parallel they
/// would race on that shared state.  Holding this lock for the entire
/// material-prep + grid-conversion section prevents the race.
static PHOTON_INIT_LOCK: std::sync::LazyLock<Mutex<()>> =
    std::sync::LazyLock::new(|| Mutex::new(()));

/// Format a duration in seconds as a short human-friendly string for
/// Redraw a fixed-height status block in place on a TTY. On the first
/// call (`*prev_lines == 0`) it just prints the lines; on later calls it
/// moves the cursor up `*prev_lines` rows and overwrites each line
/// (clearing to end-of-line first), so the block updates in place rather
/// than scrolling a fresh copy. The block grows once in practice
/// (progress-only -> progress+tally at the first checkpoint); if it ever
/// shrinks, the orphaned trailing rows are cleared so no stale text is
/// left behind.
fn render_status_block(lines: &[String], prev_lines: &mut usize) {
    let mut buf = String::new();
    if *prev_lines > 0 {
        // Move the cursor back up to the top of the previous block.
        buf.push_str(&format!("\x1b[{}A", *prev_lines));
    }
    for line in lines {
        // Return to col 0, clear the whole line, write content, newline.
        buf.push_str("\r\x1b[2K");
        buf.push_str(line);
        buf.push('\n');
    }
    // If the block shrank, clear the now-orphaned rows below, then move the
    // cursor back up so the next redraw's `prev_lines` accounting is right.
    let orphaned = prev_lines.saturating_sub(lines.len());
    for _ in 0..orphaned {
        buf.push_str("\r\x1b[2K\n");
    }
    if orphaned > 0 {
        buf.push_str(&format!("\x1b[{orphaned}A"));
    }
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(buf.as_bytes());
    let _ = out.flush();
    *prev_lines = lines.len();
}

/// Shared state for the live status panel. Written by the main thread at
/// chunk boundaries (fresh tally snapshot) and re-drawn by both the main
/// thread and the heartbeat thread -- all under one mutex so they never
/// fight over the cursor / current line.
#[derive(Default)]
struct LivePanel {
    /// Lines drawn last time (TTY/ANSI mode), so the next draw knows how
    /// far to move the cursor up.
    prev_lines: usize,
    /// Length of the last single-line draw (Jupyter `\r` mode), so a shorter
    /// update can pad away the previous line's trailing characters.
    jupyter_prev_len: usize,
    /// Per-tally value lines from the most recent chunk boundary (empty
    /// until the first boundary, or when `tally` verbosity is off).
    tally_values: Vec<String>,
    /// Transport-elapsed seconds when `tally_values` was captured, so the
    /// heartbeat can show how stale the tally snapshot is.
    checkpoint_elapsed_secs: f64,
}

/// Build the panel's lines: a progress/ETA line (always current -- `done`
/// comes from the continuously-advancing global particle counter) plus,
/// when `tally` is on, the last checkpoint's per-tally lines tagged with
/// their age.
fn build_panel_lines(
    panel: &LivePanel,
    done: u64,
    total: u64,
    elapsed_secs: f64,
    show_progress: bool,
    show_eta: bool,
    show_tally: bool,
) -> Vec<String> {
    let mut block: Vec<String> = Vec::new();
    if show_progress || show_eta {
        if total == 0 {
            // Uncapped run: no denominator, so there is no percentage or ETA
            // to show -- report the running count (with elapsed if eta is on).
            if show_eta {
                block.push(format!(
                    "Progress: {done} particles  elapsed {}",
                    format_secs(elapsed_secs),
                ));
            } else {
                block.push(format!("Progress: {done} particles"));
            }
        } else {
            let frac = done as f64 / total as f64;
            let pct = (frac * 100.0) as u64;
            if show_eta {
                let eta = if frac > 0.0 {
                    elapsed_secs * (1.0 - frac) / frac
                } else {
                    0.0
                };
                block.push(format!(
                    "Progress: {done}/{total} particles ({pct}%)  elapsed {}, ETA ~{}",
                    format_secs(elapsed_secs),
                    format_secs(eta),
                ));
            } else {
                block.push(format!("Progress: {done}/{total} particles ({pct}%)"));
            }
        }
    }
    if show_tally && !panel.tally_values.is_empty() {
        let age = (elapsed_secs - panel.checkpoint_elapsed_secs).max(0.0);
        let age_str = format_secs(age);
        for line in &panel.tally_values {
            block.push(format!("{line}   [checkpoint {age_str} ago]"));
        }
    }
    block
}

/// Redraw the live panel under the panel mutex. On a real terminal
/// (`jupyter == false`) this is a multi-line ANSI in-place block. Under a
/// Jupyter kernel (`jupyter == true`) ANSI cursor moves aren't rendered but
/// `\r` is treated as an in-place line replace, so the panel collapses to a
/// single carriage-return-updated line (padded to erase a longer previous
/// draw). No trailing newline in Jupyter mode -- the caller emits one once,
/// after the run. Called by both the boundary render and the heartbeat.
#[allow(clippy::too_many_arguments)]
fn render_live_panel(
    panel: &mut LivePanel,
    done: u64,
    total: u64,
    elapsed_secs: f64,
    show_progress: bool,
    show_eta: bool,
    show_tally: bool,
    jupyter: bool,
) {
    let block = build_panel_lines(
        panel,
        done,
        total,
        elapsed_secs,
        show_progress,
        show_eta,
        show_tally,
    );
    if jupyter {
        let line = block.join("   |   ");
        let len = line.chars().count();
        let pad = panel.jupyter_prev_len.saturating_sub(len);
        let mut out = std::io::stdout().lock();
        let _ = write!(out, "\r{line}{}", " ".repeat(pad));
        let _ = out.flush();
        panel.jupyter_prev_len = len;
    } else {
        render_status_block(&block, &mut panel.prev_lines);
    }
}

/// RAII guard that stops and joins the heartbeat thread on drop -- including
/// on early return or panic -- so the background thread can never outlive
/// the run. The thread is parked between wakeups, so it costs nothing while
/// alive (see `derive_chunk_count` for why feedback is boundary-driven).
struct HeartbeatGuard {
    stop: Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Drop for HeartbeatGuard {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// progress / ETA prints. Sub-second → ms, < 60s → "X.Xs", < 1h →
/// "Mm SSs", longer → "Hh MMm".
fn format_secs(s: f64) -> String {
    if !s.is_finite() || s < 0.0 {
        return "?".to_string();
    }
    if s < 1.0 {
        return format!("{} ms", (s * 1000.0).round() as u64);
    }
    if s < 60.0 {
        return format!("{s:.1}s");
    }
    let total = s.round() as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let sec = total % 60;
    if h > 0 {
        format!("{h}h {m:02}m")
    } else {
        format!("{m}m {sec:02}s")
    }
}

/// How charged-particle (electron) energy from photon interactions is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ElectronTreatment {
    /// Thick-target bremsstrahlung: secondary photons are produced from the
    /// electron slowing-down spectrum.
    #[default]
    Ttb,
    /// Local energy deposition: charged-particle energy is deposited at the
    /// interaction site; no bremsstrahlung photons are produced.
    Local,
}

impl ElectronTreatment {
    /// Whether thick-target bremsstrahlung is enabled.
    pub fn ttb(self) -> bool {
        matches!(self, ElectronTreatment::Ttb)
    }

    /// The lowercase string form (`"ttb"` or `"local"`).
    pub fn as_str(self) -> &'static str {
        match self {
            ElectronTreatment::Ttb => "ttb",
            ElectronTreatment::Local => "local",
        }
    }
}

impl std::str::FromStr for ElectronTreatment {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ttb" => Ok(ElectronTreatment::Ttb),
            "local" => Ok(ElectronTreatment::Local),
            other => Err(format!(
                "invalid electron_treatment {other:?}; expected \"ttb\" or \"local\""
            )),
        }
    }
}

/// `serde` default for boolean fields that default to `true` (e.g.
/// [`Model::gpu_fission_bank`]), so an older serialized model without the field
/// deserializes to the enabled state rather than `false`.
fn default_true() -> bool {
    true
}

/// Per-run execution settings for [`Model::simulate_transport`] and the other
/// run entry points ([`Model::run_with_tracking`], [`Model::transmute`], the
/// GPU dispatch).
///
/// These are properties of one *execution* -- how many histories to sample,
/// from which RNG stream, with what parallelism and time budget -- not of the
/// model. Two runs of the same model with different settings estimate the same
/// quantities (only the statistics differ) and are poolable via
/// `combine_results`; the model fingerprint accordingly never includes them.
#[derive(Debug, Clone, PartialEq)]
pub struct TransportSettings {
    /// Particle-history budget for the run, or `None` for no particle cap.
    /// When `Some(n)` the runtime internally chunks `n` into
    /// `particles_per_chunk` slices for rayon parallelism; chunking is just a
    /// progress-reporting convenience (every tally uses per-history Welford so
    /// the chunk shape doesn't affect statistics). When `None` the run has no
    /// particle limit and continues until another stop condition trips
    /// (`max_runtime` or `convergence_targets`); it is the caller's job to set
    /// one, or the run never ends. Default: `Some(1000)`.
    pub total_particles: Option<usize>,
    /// Base RNG seed. It fully determines the run: BOTH per-history streams
    /// derive from it, the source-sampling `FastRng` (via the shared
    /// `PARTICLE_SEED_STRIDE`) and the collision-physics PCG (via
    /// [`yamc_rng::history_seed`], the single definition shared
    /// with the GPU seed buffer). Two runs differing only in `seed` are
    /// therefore independent realisations of the whole simulation, so their
    /// spread is a valid run-to-run error estimate (before issue #315 the
    /// collision stream ignored the seed and only the source was re-sampled,
    /// which made that spread a large under-estimate). Give each run a distinct
    /// seed when accumulating statistics across runs with `combine_results`.
    /// Default: 1.
    pub seed: u64,
    /// Worker threads for CPU transport; `None` uses all available cores.
    /// Ignored by the GPU dispatch. Default: None.
    pub threads: Option<usize>,
    /// Optional wall-time budget in seconds. When set, the transport loop stops
    /// at the first chunk checkpoint where elapsed wall time has reached this
    /// budget, then finalizes tallies normally so the results carry valid
    /// statistics and FOM. Composes with the other stop conditions: the run
    /// ends at the first of {`total_particles` exhausted, all
    /// `convergence_targets` met, `max_runtime` elapsed}. Under MPI
    /// (`mpi_size > 1`) the stop is COLLECTIVE (each rank checks its own
    /// elapsed time; an OR-reduce agrees one global stop bit so ranks break
    /// together), as is the convergence early-stop, which gathers the per-tally
    /// moments to root, folds them and broadcasts one decision bit.
    /// Default: None.
    pub max_runtime: Option<f64>,
}

impl Default for TransportSettings {
    fn default() -> Self {
        TransportSettings {
            total_particles: Some(1000),
            seed: 1,
            threads: None,
            max_runtime: None,
        }
    }
}

/// Model is what gets serialized by `Model.save()` / `Model.load()` /
/// `model.export(html)`. Run-time results (timing stats, lost-particle
/// diagnostics) are skipped -- a Model on disk is a *specification*,
/// not a *result*. Per-run execution parameters (particle count, seed,
/// threads, time budget) are not part of the model at all -- they are
/// supplied per call via [`TransportSettings`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Model {
    pub geometry: GeometryKind,
    pub sources: Vec<ParticleSource>,
    /// Free-gas threshold multiplier for elastic scattering. Default: 400.0
    pub free_gas_threshold: f64,
    /// Whether to transport secondary photons produced by neutron
    /// interactions (coupled neutron->photon). A photon *source* always
    /// transports photons regardless of this flag -- use [`Model::has_photons`]
    /// to ask "will any photons be in flight?". Default: false
    pub transport_secondary_photons: bool,
    /// Whether the GPU transport uses a device FISSION PARTICLE BANK to follow
    /// the true fission chain (issue #78) instead of the legacy
    /// `weight *= nu_bar` + weight-cap terminator. When enabled the GPU
    /// stochastically rounds nu_bar to an integer N at each fission, continues
    /// one progeny, and banks the other N-1 to transport in subsequent
    /// generations -- the GPU twin of the CPU `sample_fission_neutrons` chain.
    /// Has NO effect on the CPU path (which always banks) and NO effect on a
    /// model with no fissile material (the fission branch is never entered, so
    /// the run is byte-identical regardless). Default: true.
    #[serde(default = "default_true")]
    pub gpu_fission_bank: bool,
    /// Energy cutoff for photons in eV. Default: 1000.0
    pub photon_cutoff_energy: f64,
    /// How electron energy from photon interactions is handled. Default: Ttb.
    pub electron_treatment: ElectronTreatment,
    /// Maximum lost particles before aborting. Default: 10
    pub max_lost_particles: usize,
    /// Hard cap on transport steps per particle on the GPU path.
    /// CPU transport terminates particles naturally (absorption /
    /// leakage / lost-particle diagnostics) and ignores this; the
    /// GPU kernel needs an explicit cap because it has no concept of
    /// "lost". Default: 100000.
    ///
    /// The default is deliberately high so the GPU matches the CPU's
    /// run-to-completion behaviour even for strong, near-pure
    /// scatterers (e.g. a pure-H2 sphere, where a 14 MeV neutron can
    /// undergo many hundreds of elastic collisions before leaking).
    /// A 1000-step cap silently truncated those histories on the GPU
    /// (~10% integral-flux deficit on H2) while the CPU ran them to
    /// completion. The loop exits as soon as a particle is absorbed or
    /// leaks, so the high cap is free for fast-escaping problems and
    /// only costs work the CPU also does for genuine moderators.
    ///
    /// A cap that binds is an error, not a warning: a GPU launch in which
    /// any history was still transporting at the cap fails the run with
    /// [`GpuDispatchError::HistoriesTruncated`](crate::gpu::GpuDispatchError),
    /// because the under-counted flux it would return is not a valid
    /// answer (fusion-neutronics/core#23). Raise the cap or run on the CPU.
    pub max_steps_per_particle: u32,
    /// Whether to use decay photons (D1S method). Default: false
    pub use_decay_photons: bool,
    /// Particle-transport tracking algorithm. Default:
    /// [`TrackingMode::Surface`] (standard distance-to-nearest-boundary).
    /// [`TrackingMode::Woodcock`] is pure delta tracking (best on dense,
    /// void-free models); [`TrackingMode::Hybrid`] is delta tracking with
    /// an automatic surface fallback in voids/low-density cells. See
    /// [`TrackingMode`] for which to pick.
    ///
    /// CPU only. The GPU kernels always surface-track: a Woodcock or Hybrid
    /// request on the GPU prints a one-line notice to stderr at every
    /// verbosity and proceeds, since the surface-tracked flux is an unbiased
    /// estimate of the same quantity (fusion-neutronics/core#23).
    #[serde(default)]
    pub tracking_mode: TrackingMode,
    /// Variance-reduction techniques applied during transport. Empty (the
    /// default) is fully analog. Techniques compose; multiplicity rules
    /// are validated at run start (at most one
    /// [`SurvivalBiasing`](crate::variance_reduction::SurvivalBiasing)
    /// entry). See [`crate::variance_reduction`].
    #[serde(default)]
    pub variance_reduction: Vec<VarianceReduction>,
    pub tallies: Vec<Arc<Tally>>,
    /// Progress output level for `simulate_transport` (reported in source
    /// particles). Default: `Stream`. See [`Verbose`] for the levels.
    #[serde(default)]
    pub verbose: Verbose,
    #[serde(skip)]
    pub last_elapsed_secs: Option<f64>,
    #[serde(skip)]
    pub last_particles_per_second: Option<u64>,
    #[serde(skip)]
    pub last_data_load_secs: Option<f64>,
    #[serde(skip)]
    pub last_transport_secs: Option<f64>,
    #[serde(skip)]
    pub lost_particles: Vec<crate::util::lost_particle::LostParticle>,
    /// Precision-based stopping criteria. When non-empty, the transport loop
    /// stops at the first checkpoint where every convergence target is
    /// satisfied. Under MPI the check is collective: the per-tally moments are
    /// gathered to root, folded with Chen's exact parallel combine, evaluated
    /// once, and the single decision bit broadcast, so a borderline target
    /// cannot flip on summation order. Runtime config; not serialized.
    ///
    /// CPU only. The GPU dispatch refuses a model carrying targets with
    /// [`GpuDispatchError::ConvergenceTargetsUnsupported`](crate::gpu::GpuDispatchError)
    /// rather than running to the particle cap while ignoring them
    /// (fusion-neutronics/core#23); GPU convergence stopping is
    /// fusion-neutronics/core#29.
    #[serde(skip)]
    pub convergence_targets: Vec<yamc_tallies::ConvergenceTarget>,
}

impl Model {
    /// True when photons will be in flight during the run -- either a
    /// source emits photons, or secondary-photon production is enabled.
    /// This (not the `transport_secondary_photons` flag alone) is what
    /// drives photon cross-section loading, so a pure photon source needs
    /// no flag. Mirrors the native auto-enable in `simulate_transport`.
    pub fn has_photons(&self) -> bool {
        self.transport_secondary_photons
            || self.use_decay_photons
            || self
                .sources
                .iter()
                .any(|s| s.particle_type() == yamc_particle::particle::ParticleType::Photon)
    }

    /// Every nuclide name referenced by the model's materials, sorted and
    /// de-duplicated. This is what decides which `<Nuclide>.arrow/` section sets
    /// a run has to fetch.
    ///
    /// Reads the flat material store rather than walking cells, so it covers
    /// mesh-backed models as well as CSG (issue #246).
    pub fn required_nuclides(&self) -> Vec<String> {
        let mut names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for mat in self.geometry.materials() {
            names.extend(mat.nuclides.keys().cloned());
        }
        names.into_iter().collect()
    }

    /// Element symbols the model needs PHOTON data for, sorted and de-duplicated.
    /// Empty when no photons will be in flight, since photon data is only loaded
    /// when [`Self::has_photons`] holds.
    ///
    /// Photon data is per element (`Fe`, not `Fe56`).
    pub fn required_elements(&self) -> Vec<String> {
        if !self.has_photons() {
            return Vec::new();
        }
        let mut names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for mat in self.geometry.materials() {
            names.extend(
                mat.nuclides
                    .keys()
                    .map(|n| yamc_element::element::element_symbol_from_nuclide(n)),
            );
        }
        names.retain(|n| !n.is_empty());
        names.into_iter().collect()
    }

    /// Create a new Model with CSG geometry.
    pub fn new(geometry: Geometry, sources: Vec<ParticleSource>, tallies: Vec<Arc<Tally>>) -> Self {
        Model {
            geometry: GeometryKind::Csg(geometry),
            sources,
            free_gas_threshold: 400.0,
            transport_secondary_photons: false,
            gpu_fission_bank: true,
            photon_cutoff_energy: 1000.0,
            electron_treatment: ElectronTreatment::Ttb,
            max_lost_particles: 10,
            max_steps_per_particle: 100_000,
            use_decay_photons: false,
            tracking_mode: TrackingMode::default(),
            variance_reduction: Vec::new(),
            tallies,
            verbose: Verbose::default(),
            last_elapsed_secs: None,
            last_particles_per_second: None,
            last_data_load_secs: None,
            last_transport_secs: None,
            lost_particles: Vec::new(),
            convergence_targets: Vec::new(),
        }
    }

    /// Create a new Model with mesh-based geometry.
    #[cfg(feature = "mesh")]
    pub fn new_with_mesh(
        mesh_geometry: crate::geometry::mesh::MeshGeometry,
        sources: Vec<ParticleSource>,
        tallies: Vec<Arc<Tally>>,
    ) -> Self {
        Model {
            geometry: GeometryKind::Mesh(Box::new(mesh_geometry)),
            sources,
            free_gas_threshold: 400.0,
            transport_secondary_photons: false,
            gpu_fission_bank: true,
            photon_cutoff_energy: 1000.0,
            electron_treatment: ElectronTreatment::Ttb,
            max_lost_particles: 10,
            max_steps_per_particle: 100_000,
            use_decay_photons: false,
            tracking_mode: TrackingMode::default(),
            variance_reduction: Vec::new(),
            tallies,
            verbose: Verbose::default(),
            last_elapsed_secs: None,
            last_particles_per_second: None,
            last_data_load_secs: None,
            last_transport_secs: None,
            lost_particles: Vec::new(),
            convergence_targets: Vec::new(),
        }
    }

    /// The survival-biasing settings, if the technique is enabled. At
    /// most one entry is allowed in `variance_reduction` (validated at
    /// run start).
    pub fn survival_biasing(&self) -> Option<&SurvivalBiasing> {
        self.variance_reduction.iter().find_map(|vr| match vr {
            VarianceReduction::SurvivalBiasing(sb) => Some(sb),
            VarianceReduction::WeightWindowBounds(_) => None,
        })
    }

    /// All weight-window maps in `variance_reduction` (empty if none).
    /// Several are legal, for example one per particle type.
    pub fn weight_windows(&self) -> Vec<&crate::variance_reduction::WeightWindowBounds> {
        self.variance_reduction
            .iter()
            .filter_map(|vr| match vr {
                VarianceReduction::WeightWindowBounds(ww) => Some(ww.as_ref()),
                VarianceReduction::SurvivalBiasing(_) => None,
            })
            .collect()
    }

    /// Sample a particle from the source(s), selecting by strength weight.
    pub fn sample_source<R: rand::Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> yamc_particle::particle::Particle {
        if self.sources.len() == 1 {
            return self.sources[0].sample(rng);
        }
        self.sample_source_with(&SourceSelector::new(&self.sources), rng)
    }

    /// Sample a particle using a [`SourceSelector`] built beforehand from
    /// this model's sources.
    ///
    /// Identical in every respect to [`Model::sample_source`] -- same RNG
    /// draws, same source for the same draw -- but the strength walk is a
    /// binary search instead of a scan, which is what makes a many-source
    /// model (a parametric plasma source is thousands of ring sources)
    /// sample in constant time per history. The selector must come from the
    /// same source list; a stale one picks the wrong source.
    pub fn sample_source_with<R: rand::Rng + ?Sized>(
        &self,
        selector: &SourceSelector,
        rng: &mut R,
    ) -> yamc_particle::particle::Particle {
        if self.sources.len() == 1 {
            return self.sources[0].sample(rng);
        }
        let threshold = rng.random::<f64>() * selector.total();
        self.sources[selector.index_for(threshold)].sample(rng)
    }

    /// Build the list of MT numbers needed for transport cross section calculation.
    /// Always includes MT 1 (total), MT 301 (heating), and MT 901 (heating-local),
    /// plus any MTs required by tallies.
    pub(crate) fn transport_mt_filter(&self) -> Vec<i32> {
        let mut mt_filter = vec![1, 301, 901];

        for tally in &self.tallies {
            for score in &tally.scores {
                if let Some(mt) = score.mt() {
                    let v = mt.as_i32();
                    if !mt_filter.contains(&v) {
                        mt_filter.push(v);
                    }
                }
            }
        }

        mt_filter
    }

    /// Run the simulation without tracking (zero overhead).
    pub fn simulate_transport(&mut self, settings: &TransportSettings) -> Result<(), String> {
        self.run_internal::<NoOpTracker>(settings, None, None, false)?;
        Ok(())
    }

    /// A clone of this model with every material's density scaled by `factor`.
    /// Cheap: macroscopic cross sections are linear in density, so the density
    /// caches are invalidated and rebuilt at run start (reusing the preserved
    /// microscopic cache) without reloading nuclear data. The original model is
    /// untouched. Used by [`Model::generate_weight_windows`] for DeGVR's
    /// reduced-density passes.
    fn scaled_density_copy(&self, factor: f64) -> Result<Model, String> {
        let mut scaled = self.clone();
        for mat_arc in scaled.geometry.materials_mut() {
            let mut mat = (**mat_arc).clone();
            match mat.density {
                Some(d) => mat.density = Some(d * factor),
                None => {
                    return Err(
                        "DeGVR requires materials with an explicit density; sum-mode / \
                         transmuted materials are not yet supported"
                            .to_string(),
                    )
                }
            }
            mat.invalidate_density_caches();
            *mat_arc = std::sync::Arc::new(mat);
        }
        Ok(scaled)
    }

    /// One flux mesh tally per requested particle: each scores a flux over the
    /// window mesh and energy groups, filtered to a single particle type. The
    /// tallies are returned in `generator.particles` order, so the fiducial and
    /// asymptotic passes line up tally `i` with particle `i`.
    ///
    /// A D1S decay-photon problem needs no special filtering here: the single
    /// photon window built from this field is applied to *every* photon in
    /// production (decay and prompt secondary alike), so building it from the
    /// combined photon flux is self-consistent, and decay and prompt gammas share
    /// the same photon cross sections and similar ~MeV energies, so their
    /// deep-penetration shape (all DeGVR extrapolates) matches. Isolating the
    /// decay field with a `parent_nuclide` filter would only reshape the tally
    /// (one bin block per parent) without improving the window.
    fn degvr_flux_tallies(
        generator: &crate::variance_reduction::WeightWindowGeneratorDeGVR,
    ) -> Result<Vec<Arc<yamc_tallies::Tally>>, String> {
        use yamc_particle::ParticleType;
        use yamc_tallies::filter::Filter;
        use yamc_tallies::{EnergyFilter, FluxScore, MeshFilter, ParticleTypeFilter, Score, Tally};
        generator
            .particles
            .iter()
            .map(|&particle| {
                let mut tally = Tally::new();
                tally.name = Some(match particle {
                    ParticleType::Neutron => "degvr_flux_neutron".to_string(),
                    ParticleType::Photon => "degvr_flux_photon".to_string(),
                });
                tally.scores = vec![Score::Flux(FluxScore)];
                let mut filters = vec![Filter::Mesh(MeshFilter::new(generator.mesh.clone()))];
                if let Some(bins) = &generator.energy_bins {
                    filters.push(Filter::Energy(EnergyFilter::new(bins.clone())));
                }
                filters.push(Filter::ParticleType(ParticleTypeFilter::new(particle)));
                tally.filters = filters;
                tally.validate()?;
                Ok(Arc::new(tally))
            })
            .collect()
    }

    /// Integrate the macroscopic total cross section along a ray, particle-free:
    /// `tau = sum over segments of Sigma_t(energy) * length`, from `start` along
    /// unit `direction` for up to `max_distance`. Void/vacuum segments (no
    /// material) contribute 0. `photon` selects the photon total attenuation
    /// (`calculate_photon_xs`) over the neutron macroscopic total (MT=1); the
    /// matching per-material XS must be built (see
    /// [`Self::estimate_density_reduction`], which prepares a probe). Used by
    /// DeGVR auto-N to estimate the problem's optical thickness.
    fn integrate_optical_depth(
        &self,
        start: [f64; 3],
        direction: [f64; 3],
        max_distance: f64,
        energy: f64,
        photon: bool,
    ) -> f64 {
        const NUDGE: f64 = 1e-8;
        const MAX_STEPS: usize = 100_000;
        let mut pos = start;
        let mut remaining = max_distance;
        let mut tau = 0.0;
        let mut cell_idx = match self.geometry.find_cell_index((pos[0], pos[1], pos[2])) {
            Some(i) => i,
            None => return 0.0,
        };
        for _ in 0..MAX_STEPS {
            let cells = self.geometry.cells();
            // Photon windows attenuate on the photon total cross section; neutron
            // windows on the neutron macroscopic total (MT=1). Using the wrong one
            // gives a physically wrong optical depth (and thus N).
            let sigma_t = self
                .geometry
                .material_for(&cells[cell_idx])
                .map(|m| {
                    if photon {
                        m.calculate_photon_xs(energy).total
                    } else {
                        m.lookup_xs_by_mt(1, energy)
                    }
                })
                .unwrap_or(0.0);
            let d = self
                .geometry
                .closest_boundary(cell_idx, pos, direction)
                .map(|h| h.distance)
                .unwrap_or(remaining);
            let seg = d.min(remaining);
            tau += sigma_t * seg;
            remaining -= seg;
            if remaining <= NUDGE {
                break;
            }
            pos = [
                pos[0] + direction[0] * (d + NUDGE),
                pos[1] + direction[1] * (d + NUDGE),
                pos[2] + direction[2] * (d + NUDGE),
            ];
            cell_idx = match self.geometry.find_cell_index((pos[0], pos[1], pos[2])) {
                Some(i) => i,
                None => break, // left the geometry (vacuum)
            };
        }
        tau
    }

    /// A representative source point, energy, and particle type for the auto-N
    /// ray-trace: the dominant source (max strength), with its spatial + energy
    /// distributions sampled under a fixed seed. Exact for a monoenergetic point
    /// source; a representative draw for distributed / spectrum sources. The
    /// particle type distinguishes a photon *source* (whose energy is the
    /// representative photon energy) from a neutron source that only *drives*
    /// secondary / decay photons.
    fn representative_source(
        &self,
    ) -> Result<([f64; 3], f64, yamc_particle::ParticleType), String> {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let src = self
            .sources
            .iter()
            .max_by(|a, b| {
                a.strength()
                    .partial_cmp(&b.strength())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or("DeGVR auto density_reduction requires at least one source")?;
        let s = src.source();
        let mut rng = StdRng::seed_from_u64(0x0DE6_2023);
        let pos = s.space.sample(&mut rng);
        // Representative energy: mean of several draws (exact when monoenergetic).
        let e = (0..64).map(|_| s.energy.sample(&mut rng)).sum::<f64>() / 64.0;
        Ok((pos, e, src.particle_type()))
    }

    /// A high percentile of the per-voxel optical depth from `src_pos` to each
    /// window-mesh voxel center, integrated at `energy` using photon attenuation
    /// (`photon`) or the neutron macroscopic total (MT=1). `self` must already
    /// carry original-density macroscopic XS (and photon element data when
    /// `photon`). Drives the auto DeGVR N via [`WeightWindowGeneratorDeGVR::n_from_tau`].
    fn optical_depth_percentile(
        &self,
        src_pos: [f64; 3],
        mesh: &yamc_tallies::RegularRectangularMesh,
        energy: f64,
        photon: bool,
    ) -> f64 {
        let ll = mesh.lower_left();
        let w = mesh.width();
        let [nx, ny, nz] = mesh.shape();
        let nv = nx * ny * nz;
        // Cap candidate targets on very fine meshes (a strided scan still spans
        // the depth distribution).
        let stride = (nv / 30_000).max(1);
        let mut taus: Vec<f64> = Vec::new();
        let mut flat = 0usize;
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let take = flat.is_multiple_of(stride);
                    flat += 1;
                    if !take {
                        continue;
                    }
                    let center = [
                        ll[0] + (ix as f64 + 0.5) * w[0],
                        ll[1] + (iy as f64 + 0.5) * w[1],
                        ll[2] + (iz as f64 + 0.5) * w[2],
                    ];
                    let dv = [
                        center[0] - src_pos[0],
                        center[1] - src_pos[1],
                        center[2] - src_pos[2],
                    ];
                    let l = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
                    if l <= 0.0 {
                        continue;
                    }
                    let dir = [dv[0] / l, dv[1] / l, dv[2] / l];
                    taus.push(self.integrate_optical_depth(src_pos, dir, l, energy, photon));
                }
            }
        }
        // Use a high percentile of the per-voxel optical depth, not the raw max.
        // For a box shield the single deepest voxel is a far corner reached by a
        // grazing oblique ray through several wall-thicknesses, which
        // over-estimates the depth that should set the windows. A percentile
        // rejects that rare corner while preserving a sphere's radial edge (a
        // large fraction of a cubic mesh's voxels sit at the sphere surface / in
        // the surrounding shell, all at the full radial depth).
        const DEPTH_PERCENTILE: f64 = 0.90;
        taus.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if taus.is_empty() {
            0.0
        } else {
            let idx = (((taus.len() - 1) as f64) * DEPTH_PERCENTILE).round() as usize;
            taus[idx]
        }
    }

    /// Estimate the DeGVR density-reduction factor `N` from the problem's optical
    /// thickness via a particle-free ray-trace from the source to the deepest
    /// mesh voxel. With several requested particles the per-species estimates are
    /// combined by taking the **max** N (thinnest fiducial), so one reduced-density
    /// pair seeds every field. Returns `(n, tau)` for the species that set the max.
    /// The current model is not mutated (works on a probe clone).
    ///
    /// A photon window's ray-trace integrates the photon attenuation at a
    /// *photon* energy: the source energy when photons are emitted by a photon
    /// source, else the `photon_energy` override or a representative secondary /
    /// decay gamma energy. It must not use the driving neutron energy (14 MeV
    /// attenuation is unlike ~MeV-gamma attenuation). Neutron windows use the
    /// neutron source energy and MT=1.
    pub fn estimate_density_reduction(
        &self,
        generator: &crate::variance_reduction::WeightWindowGeneratorDeGVR,
    ) -> Result<(f64, f64), String> {
        use yamc_particle::ParticleType;
        let (src_pos, src_e, src_ptype) = self.representative_source()?;
        let wants_photon = generator.particles.contains(&ParticleType::Photon);
        // Probe carrying original-density macroscopic XS. If any photon window is
        // requested we need the photon attenuation, so keep secondary-photon
        // transport on: `prepare_transport_data` then builds the per-material
        // photon element data that `calculate_photon_xs` reads. Otherwise MT=1
        // suffices and photon init is skipped.
        let mut probe = self.clone();
        probe.transport_secondary_photons = wants_photon;
        probe.variance_reduction = Vec::new();
        probe.prepare_transport_data(&[1], false, false)?;

        let mut best: Option<(f64, f64)> = None;
        for &particle in &generator.particles {
            let photon = particle == ParticleType::Photon;
            let energy = if photon {
                generator.photon_energy.unwrap_or({
                    if src_ptype == ParticleType::Photon {
                        src_e
                    } else {
                        crate::variance_reduction::DEGVR_PHOTON_REP_ENERGY_EV
                    }
                })
            } else {
                src_e
            };
            let tau = probe.optical_depth_percentile(src_pos, &generator.mesh, energy, photon);
            let n = crate::variance_reduction::WeightWindowGeneratorDeGVR::n_from_tau(tau);
            if best.is_none_or(|(bn, _)| n > bn) {
                best = Some((n, tau));
            }
        }
        best.ok_or_else(|| {
            "WeightWindowGeneratorDeGVR requires at least one particle type".to_string()
        })
    }

    /// Generate weight windows with the DeGVR method (density-extrapolation).
    ///
    /// Runs two reduced-density fixed-source passes internally: a low-density
    /// fiducial pass (all densities / N, analog) and a half-reduction
    /// asymptotic pass (all densities / (N / 2)) that applies the fiducial
    /// windows so it reaches the whole mesh. It extrapolates the two flux
    /// fields back to the original density and returns one
    /// [`crate::variance_reduction::WeightWindowBounds`] per requested particle
    /// (in `generator.particles` order) for a clean production run. `settings`
    /// controls the (throwaway) generation passes. The current model is not
    /// mutated.
    ///
    /// For a coupled request (`particles = [Neutron, Photon]`) both windows come
    /// from the **same** pair of passes: the fiducial pass tallies one flux mesh
    /// per particle and builds a fiducial window each; the asymptotic pass
    /// applies *all* the fiducial windows at once and re-tallies every field.
    /// A single N (the max over species when auto) keeps the fiducial thin enough
    /// to seed each field. The photon field captures secondary photons (and, in a
    /// D1S run, decay photons) transported alongside the neutrons.
    pub fn generate_weight_windows(
        &self,
        generator: &crate::variance_reduction::WeightWindowGeneratorDeGVR,
        settings: &TransportSettings,
    ) -> Result<Vec<crate::variance_reduction::WeightWindowBounds>, String> {
        use crate::variance_reduction::{VarianceReduction, WeightWindowBounds};
        generator.validate()?;
        // Resolve N (one N for every species): explicit override, or auto from a
        // particle-free optical-depth ray-trace (max over species).
        let n = match generator.density_reduction {
            Some(explicit) => explicit,
            None => {
                let (n, tau) = self.estimate_density_reduction(generator)?;
                eprintln!(
                    "DeGVR auto density_reduction: N = {n:.2} (estimated optical depth tau = {tau:.2})"
                );
                n
            }
        };
        let expected = generator.n_groups() * generator.mesh.num_voxels();
        let particles = &generator.particles;

        // Fiducial pass: all densities / N, analog; one flux mesh per particle.
        let mut fiducial = self.scaled_density_copy(1.0 / n)?;
        fiducial.tallies = Self::degvr_flux_tallies(generator)?;
        fiducial.variance_reduction = Vec::new();
        fiducial.simulate_transport(settings)?;
        let mut fiducial_fields: Vec<Vec<f64>> = Vec::with_capacity(particles.len());
        let mut fiducial_windows: Vec<WeightWindowBounds> = Vec::with_capacity(particles.len());
        for (i, &particle) in particles.iter().enumerate() {
            let f = fiducial.tallies[i].get_mean();
            if f.len() != expected {
                return Err(format!(
                    "DeGVR fiducial {particle:?} flux tally returned {} bins, expected {expected}",
                    f.len()
                ));
            }
            fiducial_windows.push(generator.build_bounds(&f, particle));
            fiducial_fields.push(f);
        }

        // Asymptotic pass: all densities / (N/2), applying every fiducial window.
        let mut asymptotic = self.scaled_density_copy(2.0 / n)?;
        asymptotic.tallies = Self::degvr_flux_tallies(generator)?;
        asymptotic.variance_reduction = fiducial_windows
            .into_iter()
            .map(|w| VarianceReduction::WeightWindowBounds(Box::new(w)))
            .collect();
        asymptotic.simulate_transport(settings)?;

        // Extrapolate each field to the original density and build each window.
        let mut windows = Vec::with_capacity(particles.len());
        for (i, &particle) in particles.iter().enumerate() {
            let h = asymptotic.tallies[i].get_mean();
            if h.len() != expected {
                return Err(format!(
                    "DeGVR asymptotic {particle:?} flux tally returned {} bins, expected {expected}",
                    h.len()
                ));
            }
            let g = generator.extrapolate(&fiducial_fields[i], &h, n);
            windows.push(generator.build_bounds(&g, particle));
        }
        Ok(windows)
    }

    /// Populate the global LINEAR-space TTB energy grids from element 0's raw
    /// data, if a cached element exposes them. The grids are set once and
    /// never mutated in place: `init_bremsstrahlung` reads the linear grid and
    /// the separate log grid is built afterwards, so a concurrent photon
    /// simulation's transport can't read a grid that's mid-reset here.
    ///
    /// Callers guard this with their own "TTB enabled" / "photon transport"
    /// condition; this helper performs the population unconditionally.
    fn ensure_global_ttb_grids() {
        if let Some(elem) = yamc_element::photon::get_element_by_index(0) {
            if !elem.ttb_electron_energy.is_empty() {
                yamc_element::photon::set_ttb_e_grid(elem.ttb_electron_energy.clone());
            }
            if !elem.ttb_photon_energy.is_empty() {
                yamc_element::photon::set_ttb_k_grid(elem.ttb_photon_energy.clone());
            }
        }
    }

    /// Shared per-material photon-data prep: `init_photon_data` (returning the
    /// upfront-presence message on failure) and, when `ttb` is set,
    /// `init_bremsstrahlung`. `slot` is the material slot index, used only in
    /// the error message.
    ///
    /// When `populate_atoms_cache` is set (the GPU path, which skips the CPU
    /// `calculate_macroscopic_xs` that would otherwise fill it), the
    /// `cached_atoms_per_barn_cm` cache that `init_bremsstrahlung` needs is
    /// populated from the composition between `init_photon_data` and
    /// `init_bremsstrahlung`.
    fn prepare_material_photon_data(
        material: &mut yamc_materials::material::Material,
        slot: usize,
        ttb: bool,
        populate_atoms_cache: bool,
    ) -> Result<(), String> {
        if let Err(e) = material.init_photon_data(&material.photon_data_paths.clone()) {
            return Err(format!(
                "Failed to initialize photon data for material slot {slot}: {e}"
            ));
        }
        if ttb {
            if populate_atoms_cache && material.cached_atoms_per_barn_cm.is_none() {
                // `init_bremsstrahlung` needs `cached_atoms_per_barn_cm`
                // (CPU prep populates it via `calculate_macroscopic_xs`).
                // GPU path skips that, so populate the cache manually
                // from the composition.
                material.cached_atoms_per_barn_cm = Some(material.get_atoms_per_barn_cm()?);
            }
            material.init_bremsstrahlung();
        }
        Ok(())
    }

    /// Walk every material and run the minimal photon prep that the
    /// GPU photon path needs -- `ensure_nuclides_loaded` +
    /// `init_photon_data` and, when `electron_treatment` is `Ttb`,
    /// `init_bremsstrahlung`. The CPU path runs this inline inside
    /// `run_internal`; the GPU dispatch (`run_on_gpu` /
    /// `run_on_gpu_photon`) skips that block and would otherwise see
    /// materials with empty `cached_elements` (and, with TTB enabled,
    /// no bremsstrahlung tables), which makes
    /// `extract_photon_material_xs` panic on the first launch of a
    /// fresh model.
    ///
    /// Idempotent: re-runs `init_photon_data` even when
    /// `cached_elements` is already populated (the method itself
    /// clears + repopulates), so it's safe to call before every GPU
    /// launch.
    ///
    /// Returns the same upfront-presence error message as the CPU path
    /// if `transport_secondary_photons=true` but any material's
    /// `photon_data_paths` is empty.
    pub fn ensure_photon_data_for_gpu(&mut self) -> Result<(), String> {
        // Gate on `has_photons()`, not `transport_secondary_photons`: a pure
        // photon source needs the same TTB / Doppler / relaxation prep as
        // the coupled mode. Gating on the coupling flag alone left the GPU's
        // TTB tables empty for photon-source models, silencing the entire
        // bremsstrahlung photon source the CPU emits -- the cause of issue
        // #415's photoelectric ~0.38x deficit (the CPU's low-energy photon
        // spectrum comes from TTB photons that never existed on the GPU).
        if !self.has_photons() {
            return Ok(());
        }

        // Upfront presence check -- mirrors the CPU-path check at the
        // top of `run_internal`. Surfacing the missing-data error here
        // means the user sees a clean message instead of a panic
        // inside `extract_photon_material_xs`.
        for (i, material_arc) in self.geometry.materials().iter().enumerate() {
            if !material_arc.nuclides.is_empty() && material_arc.photon_data_paths.is_empty() {
                return Err(format!(
                    "photon transport requested but material slot {i} has no photon data. \
                     Pass `photon_data` to `Material.read_nuclear_data` (e.g. \
                     `read_nuclear_data('endf-b8.1')` populates it from the keyword), \
                     or supply a per-element map: \
                     `read_nuclear_data(..., photon_data={{'Fe': '/path/to/Fe.arrow'}})`."
                ));
            }
        }

        // Serialise photon-init with any concurrent simulate_transport
        // calls -- they both touch the global TTB grids.
        let _guard = PHOTON_INIT_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let ttb_enabled = self.electron_treatment.ttb();

        // Ensure the global TTB grids are populated from element 0's raw
        // (LINEAR-space) data if some earlier load hasn't already set
        // them. The grids are set once and never mutated in place;
        // `init_bremsstrahlung` reads the linear grid and the separate
        // log grid is built below, so concurrent simulations don't race
        // on a grid being re-cleared / re-converted under them.
        if ttb_enabled {
            Self::ensure_global_ttb_grids();
        }

        for (i, material_arc) in self.geometry.materials_mut().iter_mut().enumerate() {
            let mut material = (**material_arc).clone();
            let _ = material.ensure_nuclides_loaded();
            Self::prepare_material_photon_data(&mut material, i, ttb_enabled, true)?;
            *material_arc = Arc::new(material);
        }

        if ttb_enabled {
            yamc_element::photon::ensure_ttb_e_grid_log();
        }

        Ok(())
    }

    /// Widen every material's nuclide data to cover the temperature it will be
    /// extracted at, which the GPU dispatch needs and the CPU gets for free.
    ///
    /// `read_nuclear_data` narrows the load to the material's temperature at the
    /// time it runs, so relabelling afterwards (`material.temperature = 900`
    /// from Python) leaves the label naming reactions that were never parsed in.
    /// The CPU recovers because `run_internal` calls `calculate_macroscopic_xs`,
    /// which widens on the way to building the grid. The GPU dispatch skips that
    /// block, so it reaches `extract_material_xs` with the narrow data and fails
    /// with `TemperatureNotLoaded` (#481).
    ///
    /// An empty label is resolved rather than skipped. The CPU resolves it in
    /// `calculate_macroscopic_xs` via `resolve_temperature`, which adopts the
    /// single available temperature; nothing on the GPU path calls that, so
    /// skipping here would leave `extract_material_xs` looking up the empty
    /// string and failing in exactly the way this function exists to prevent.
    ///
    /// Runs before the photon prep, matching the CPU's order, and returns
    /// without cloning when nothing needs widening.
    pub fn ensure_neutron_temperatures_for_gpu(&mut self) -> Result<(), String> {
        for (slot, material_arc) in self.geometry.materials_mut().iter_mut().enumerate() {
            if material_arc.nuclide_data.is_empty() {
                continue;
            }
            let labelled = !material_arc.temperature().is_empty();
            let temperature = if labelled {
                material_arc.temperature().to_string()
            } else {
                // Same fallback `resolve_temperature` uses: the single
                // temperature every nuclide agrees on. Left alone when there is
                // no single answer, so the existing error still names the
                // problem rather than this function guessing.
                let mut all: Vec<String> = material_arc
                    .nuclide_data
                    .values()
                    .flat_map(|n| n.available_temperatures.iter().cloned())
                    .collect();
                all.sort();
                all.dedup();
                match all.len() {
                    1 => all.remove(0),
                    _ => continue,
                }
            };

            // Servable, not merely listed. The GPU cannot blend in-kernel: its
            // extractors do an exact `get_temp_idx` match and return
            // `TemperatureNotLoaded` on a miss. Because the blend is
            // materialised on the host as a real `loaded_temperatures` entry,
            // widening here is the only GPU-side change interpolation needs,
            // and both backends then read the same arrays by construction.
            let needs_widening = material_arc.nuclide_data.values().any(|n| {
                !n.loaded_temperatures.contains(&temperature)
                    && yamc_nuclide::temperature::resolve(&temperature, &n.available_temperatures)
                        .is_ok()
            });
            if !needs_widening && labelled {
                continue;
            }

            let mut material = (**material_arc).clone();
            if !labelled {
                material.set_temperature(&temperature);
            }
            material
                .ensure_temperature_loaded(&temperature)
                .map_err(|e| {
                    format!(
                        "Material slot {slot}: could not load temperature \
                         '{temperature}' for the GPU dispatch: {e}"
                    )
                })?;
            *material_arc = Arc::new(material);
        }
        Ok(())
    }

    /// Run the simulation with particle tracking enabled.
    ///
    /// This captures detailed event-by-event data during transport for debugging
    /// and analysis. The tracking data is returned as a TrackStorage object.
    ///
    /// # Arguments
    /// * `settings` - Per-run execution settings (see [`TransportSettings`])
    /// * `selection` - Which global history indices to capture (see
    ///   [`HistorySelection`])
    ///
    /// # Returns
    /// TrackStorage containing all tracked particle events
    pub fn run_with_tracking(
        &mut self,
        settings: &TransportSettings,
        selection: HistorySelection,
    ) -> Result<TrackStorage, String> {
        self.run_internal::<RealTracker>(settings, Some(selection), None, true)
    }

    /// Load and prepare per-material transport data before the transport
    /// loop: validate photon-data presence, acquire the photon-init lock,
    /// set up the LINEAR-space TTB grids, compute macroscopic / per-nuclide
    /// / photon / bremsstrahlung XS per material, build the LOG-space TTB
    /// grid, release the lock, and return the data-load wall time.
    fn prepare_transport_data(
        &mut self,
        mt_filter: &[i32],
        has_nuclide_tallies: bool,
        debug: bool,
    ) -> Result<f64, String> {
        #[cfg(feature = "debug_timing")]
        let material_prep_start = std::time::Instant::now();

        // Validate photon data presence BEFORE acquiring the photon-init lock,
        // so that an explicit user error (forgot photon_data) doesn't poison
        // the global mutex and contaminate subsequent simulations in the same
        // process. See `init_photon_data` for the per-element check.
        if self.transport_secondary_photons {
            for (i, material_arc) in self.geometry.materials().iter().enumerate() {
                if !material_arc.nuclides.is_empty() && material_arc.photon_data_paths.is_empty() {
                    return Err(format!(
                        "transport_secondary_photons=True but material slot {i} has no photon data. \
                         Pass `photon_data` to `Material.read_nuclear_data` (e.g. \
                         `read_nuclear_data('endf-b8.1')` populates it from the keyword), \
                         or supply a per-element map: \
                         `read_nuclear_data(..., photon_data={{'Fe': '/path/to/Fe.arrow'}})`."
                    ));
                }
            }
        }

        // Acquire the photon-init lock when photon transport is enabled.
        // The global TTB grids go through a LINEAR → LOG conversion and
        // concurrent tests would race on that shared state without this.
        let _photon_guard = if self.transport_secondary_photons {
            Some(
                PHOTON_INIT_LOCK
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            )
        } else {
            None
        };

        // Ensure global TTB grids are populated (LINEAR space) from any
        // cached element's raw data. Set once and never mutated in place:
        // init_bremsstrahlung reads the linear grid and the separate log
        // grid is built after the material loop, so a concurrent photon
        // simulation's transport can't read a grid that's mid-reset here.
        if self.transport_secondary_photons {
            Self::ensure_global_ttb_grids();
        }

        let data_load_start = crate::util::timer::Timer::start();
        let transport_secondary_photons = self.transport_secondary_photons;
        let ttb = self.electron_treatment.ttb();
        // `calculate_macroscopic_xs` borrows a `&Vec<i32>`; own the slice
        // once here so the per-material loop borrows it without reallocating.
        let mt_filter = mt_filter.to_vec();
        for (i, material_arc) in self.geometry.materials_mut().iter_mut().enumerate() {
            // Skip materials that already have transport XS computed (e.g., from transmutation step)
            if material_arc.fast_xs.is_some() {
                if debug {
                    eprintln!("[DEBUG] Material slot {i} already has fast_xs, skipping...");
                }
                continue;
            }
            if debug {
                eprintln!("[DEBUG] Material slot {i} calculating macroscopic XS...");
            }
            let mut material = (**material_arc).clone();
            let _ = material.ensure_nuclides_loaded();
            material.calculate_macroscopic_xs(&mt_filter, true);
            if has_nuclide_tallies {
                material.populate_per_nuclide_xs();
            }
            if transport_secondary_photons {
                // The presence check is performed upfront before any locks
                // are taken; init_photon_data will surface per-element issues.
                Self::prepare_material_photon_data(&mut material, i, ttb, false)?;
            }
            *material_arc = Arc::new(material);
        }

        // After all materials are initialized, build the log-space TTB
        // energy grid for transport-time interpolation (idempotent;
        // leaves the linear grid intact).
        if self.transport_secondary_photons {
            yamc_element::photon::ensure_ttb_e_grid_log();
        }

        // Release the photon-init lock (guard dropped here)
        drop(_photon_guard);

        let data_load_secs = data_load_start.elapsed_secs();

        #[cfg(feature = "debug_timing")]
        eprintln!(
            "[TIMING] run_internal material prep: {:.3}s",
            material_prep_start.elapsed().as_secs_f64()
        );

        Ok(data_load_secs)
    }

    /// Validate the variance-reduction configuration before any expensive work:
    /// per-technique weight parameters, and at most one `SurvivalBiasing` entry
    /// (a second is meaningless).
    fn validate_variance_reduction(&self) -> Result<(), String> {
        let mut survival_count = 0usize;
        for vr in &self.variance_reduction {
            match vr {
                VarianceReduction::SurvivalBiasing(sb) => {
                    survival_count += 1;
                    sb.validate()?;
                }
                VarianceReduction::WeightWindowBounds(ww) => {
                    ww.validate()?;
                }
            }
        }
        if survival_count > 1 {
            return Err(format!(
                "variance_reduction may contain at most one SurvivalBiasing entry (got {survival_count})"
            ));
        }
        Ok(())
    }

    /// Build the neutron Woodcock majorant: a global Σ_t bound over *all*
    /// materials (a per-material bound is invalid along flights that cross into
    /// a denser material). `None` for surface tracking, which needs no majorant.
    fn build_majorant(&self) -> Option<std::sync::Arc<dyn yamc_materials::Majorant>> {
        if self.tracking_mode != TrackingMode::Surface {
            let material_refs: Vec<&yamc_materials::Material> = self
                .geometry
                .materials()
                .iter()
                .map(|m| m.as_ref())
                .collect();
            Some(std::sync::Arc::new(yamc_materials::GlobalMajorant::new(
                &material_refs,
            )))
        } else {
            None
        }
    }

    /// Build the photon Woodcock majorant when photons can appear (photon
    /// sources, coupled production, or D1S decay -- all of which have set
    /// `transport_secondary_photons` by this point) and we're not surface
    /// tracking; bounds the macroscopic photon total XS across the model.
    fn build_photon_majorant(&self) -> Option<std::sync::Arc<dyn yamc_materials::Majorant>> {
        if self.tracking_mode != TrackingMode::Surface && self.transport_secondary_photons {
            let material_refs: Vec<&yamc_materials::Material> = self
                .geometry
                .materials()
                .iter()
                .map(|m| m.as_ref())
                .collect();
            Some(std::sync::Arc::new(
                yamc_materials::GlobalPhotonMajorant::new(&material_refs),
            ))
        } else {
            None
        }
    }

    /// Load the D1S chain file and precompute decay-photon data when
    /// `use_decay_photons` is set (otherwise an empty Vec). Interns nuclide
    /// names into `nuclide_registry` and re-Arcs the model's materials with
    /// `sorted_nuclide_ids` populated so the hot path avoids String lookups.
    fn prepare_decay_photon_data(
        &mut self,
        nuclide_registry: &mut yamc_nuclide::nuclide_registry::NuclideRegistry,
        is_root: bool,
    ) -> Result<
        Vec<Vec<yamc_physics::photon::decay_photon_production::DecayPhotonNuclideData>>,
        String,
    > {
        if !self.use_decay_photons {
            return Ok(Vec::new());
        }
        // Assemble the transmutation chain from the configured per-subsection
        // sources (decay / reactions / fission_yields), each resolved and
        // downloaded independently. Decay data (needed for decay-photon
        // sources) defaults to the endf-b8.1 library when unset.
        let chain = yani_transmute::load_configured_chain()
            .map_err(|e| {
                format!(
                    "Failed to load transmutation chain for decay photons: {}",
                    e
                )
            })?
            .chain;
        let mut all_nuclides: HashMap<String, Arc<yamc_nuclide::nuclide::Nuclide>> = HashMap::new();
        for mat in self.geometry.materials() {
            for (name, nuclide) in &mat.nuclide_data {
                all_nuclides
                    .entry(name.clone())
                    .or_insert_with(|| nuclide.clone());
            }
        }
        let data = yamc_physics::photon::decay_photon_production::precompute_decay_photon_data(
            &chain,
            &all_nuclides,
            nuclide_registry,
        );
        if is_root && self.verbose.summary {
            let n_nuclides = data.iter().filter(|v| !v.is_empty()).count();
            let n_channels: usize = data
                .iter()
                .flat_map(|v| v.iter())
                .map(|d| d.channels.len())
                .sum();
            println!(
                "D1S: precomputed photon data for {n_nuclides} nuclides ({n_channels} channels)"
            );
        }

        // Second pass over materials: clone, populate sorted_nuclide_ids
        // from the now-interned registry, and re-Arc so the hot path can read
        // NuclideIds from collision_data without a String lookup.
        for material_arc in self.geometry.materials_mut().iter_mut() {
            let mut material = (**material_arc).clone();
            material.populate_sorted_nuclide_ids(nuclide_registry);
            *material_arc = Arc::new(material);
        }

        Ok(data)
    }

    /// Prepare D1S decay-photon data for the GPU dispatch and resolve any
    /// `parent_nuclides` tally filters against the SAME freshly-built registry.
    ///
    /// This mirrors the registry setup `run_internal` does (build one registry,
    /// precompute decay data interning the chain emitters, then resolve every
    /// tally's `ParentNuclideFilter` against it). Returning the registry lets
    /// the GPU dispatch read back the channel `target_id`s and the filter bin
    /// ids from one consistent id space -- the parent-nuclide tag the GPU
    /// stamps on each decay photon then agrees with the filter `get_bin`.
    ///
    /// No-op data (empty Vec) and an empty registry are returned when
    /// `use_decay_photons` is false.
    // Only the coupled GPU dispatch (`run_on_gpu_coupled`) calls this, and that
    // path is `cfg(not(target_os = "macos"))`; gate this to match, else it is
    // dead code under `-D warnings` on the macOS gpu build.
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    pub(crate) fn prepare_decay_photon_data_for_gpu(
        &mut self,
    ) -> Result<
        (
            Vec<Vec<yamc_physics::photon::decay_photon_production::DecayPhotonNuclideData>>,
            yamc_nuclide::nuclide_registry::NuclideRegistry,
        ),
        String,
    > {
        let mut registry = yamc_nuclide::nuclide_registry::NuclideRegistry::new();
        let data = self.prepare_decay_photon_data(&mut registry, false)?;
        for tally_arc in &self.tallies {
            if let Some(pnf) = tally_arc.get_parent_nuclide_filter() {
                pnf.resolve(&mut registry);
            }
        }
        Ok((data, registry))
    }

    /// Internal generic transport loop - monomorphized for each tracker type.
    /// When T=NoOpTracker, all tracking calls optimize to nothing.
    pub(crate) fn run_internal<T: Tracker + Clone>(
        &mut self,
        settings: &TransportSettings,
        selection: Option<HistorySelection>,
        transmutation_tallies: Option<Arc<TransmutationTallies>>,
        print_tracking_summary: bool,
    ) -> Result<TrackStorage, String> {
        let threads = settings.threads;
        let start_time = crate::util::timer::Timer::start();
        let debug = std::env::var("YAMC_DEBUG").is_ok();

        // Validate the variance-reduction configuration before any expensive work.
        self.validate_variance_reduction()?;

        // Mesh-filled cells require surface tracking: Woodcock/hybrid
        // flights cross cells against a global majorant and never see the
        // embedded mesh surfaces, so results would be silently biased.
        if self.tracking_mode != TrackingMode::Surface && self.geometry.has_mesh_fills() {
            return Err(format!(
                "tracking_mode='{}' does not support mesh-filled cells: delta \
                 tracking never sees the embedded mesh surfaces. Use \
                 tracking_mode='surface'.",
                self.tracking_mode
            ));
        }

        // Auto-enable transport_secondary_photons if any source emits photons
        if self
            .sources
            .iter()
            .any(|s| s.particle_type() == yamc_particle::particle::ParticleType::Photon)
        {
            self.transport_secondary_photons = true;
        }

        // Initialize MPI context (no-op if MPI not compiled)
        let mpi_ctx = MpiContext::init();
        let mpi_rank = mpi_ctx.rank();
        let mpi_size = mpi_ctx.size();
        let is_root = mpi_ctx.is_root();

        // Only root prints MPI info
        if is_root && mpi_size > 1 {
            println!("Running with MPI: {mpi_size} ranks");
        }

        // An uncapped run ends only on `max_runtime` or convergence. Under MPI
        // both now stop collectively: `max_runtime` through the OR-reduced stop
        // bit below, and convergence through the gather-fold-broadcast at the
        // chunk checkpoint. So the only combination left with no stop condition
        // at all is uncapped, untimed AND with no convergence targets set.
        // Reject that up front rather than hang.
        if settings.total_particles.is_none()
            && settings.max_runtime.is_none()
            && self.convergence_targets.is_empty()
            && mpi_size > 1
        {
            return Err(
                "An uncapped run (total_particles = None) with no max_runtime and no convergence \
                 targets has no stop condition and would never terminate. Set total_particles, \
                 max_runtime, or convergence targets for MPI runs."
                    .to_string(),
            );
        }

        // Derive chunk size for progress reporting (defaults to
        // total_particles/10 so the user sees ~10 progress lines).
        // Chunking doesn't affect statistics: every tally uses
        // per-history Welford.
        // Front-loaded chunk schedule: early boundaries at 100, 1_000,
        // 10_000, ... particles, then ~10 evenly-spaced boundaries to the
        // end -- so ETA/tally/progress appear almost immediately on a large
        // run, then at a steady cadence. Chunk shape never affects results
        // (per-history Welford; seeds are keyed to the global particle
        // index), and the schedule is derived from `total_particles` alone
        // so every MPI rank computes the identical one. `mpi_decompose` now
        // runs per chunk inside the loop, since chunk sizes vary.
        //
        // A capped run (`Some(total)`) splits `total` into `derive_chunk_count`
        // chunks whose sizes come lazily from `derive_chunk_size` (bounded by
        // `MAX_PARTICLES_PER_CPU_CHUNK` so a huge total never materialises a
        // giant schedule, issue #193). An uncapped run (`None`) has no finite
        // count: the batch loop instead draws sizes from `uncapped_chunk_size`
        // and runs until `max_runtime`/convergence trips. `num_chunks` (used
        // only to seed the tally batch counter, which finalize overwrites with
        // the real realization count) is 0 for an uncapped run.
        let num_chunks = match settings.total_particles {
            Some(total) => derive_chunk_count(total),
            None => 0,
        };

        if debug || (is_root && mpi_size > 1) {
            let (rank_particles, rank_offset) =
                mpi_decompose(settings.total_particles.unwrap_or(0), mpi_rank, mpi_size);
            eprintln!(
                "[MPI Rank {mpi_rank}] ~{rank_particles} particles \
                 (indices {rank_offset}..{}) across {num_chunks} chunks",
                rank_offset + rank_particles,
            );
        }

        // Prepare an optional local Rayon thread pool if a specific thread count is requested
        let local_pool =
            threads.and_then(|n| rayon::ThreadPoolBuilder::new().num_threads(n).build().ok());

        // Ensure all nuclear data is loaded before transport
        if debug {
            eprintln!(
                "[DEBUG] Preparing materials: {} cells in geometry",
                self.geometry.num_cells()
            );
        }
        let mt_filter = self.transport_mt_filter();

        // Check if any tally uses per-nuclide scoring
        let has_nuclide_tallies = self.tallies.iter().any(|t| !t.nuclides.is_empty());

        // Log each nuclear-data file as it loads when verbose "nuclear_data" is
        // set. Scoped to the data-load phase; reset even if loading errors so
        // the process-global toggle never leaks past this run.
        if self.verbose.nuclear_data {
            yamc_nuclide::set_load_logging(true);
        }
        let prepared = self.prepare_transport_data(&mt_filter, has_nuclide_tallies, debug);
        if self.verbose.nuclear_data {
            yamc_nuclide::set_load_logging(false);
        }
        let data_load_secs = prepared?;

        // Ensure all tallies are initialized with the correct number
        // of batches. The runtime now derives this from the tally
        // configuration (see `derive_chunk_count`).
        for tally_arc in &self.tallies {
            tally_arc.initialize_batches_shared(num_chunks);
        }

        // Validate every tally (filter consistency, multiply_density
        // requirements, score/estimator compatibility -- see
        // `Tally::validate`). Validation is O(scores+filters) per tally
        // and only runs once here, so the cost is negligible.
        for tally_arc in &self.tallies {
            if let Err(e) = tally_arc.validate() {
                return Err(format!("Tally validation error: {e}"));
            }
        }

        // Prepare overlay XS for tallies with multiply_density=false.
        // Collect photon data paths from all materials for photon overlay.
        let has_overlay_tallies = self.tallies.iter().any(|t| !t.multiply_density);
        if has_overlay_tallies {
            let mut all_photon_paths: HashMap<String, String> = HashMap::new();
            for mat in self.geometry.materials() {
                for (elem, path) in &mat.photon_data_paths {
                    all_photon_paths
                        .entry(elem.clone())
                        .or_insert_with(|| path.clone());
                }
            }
            // Which species can actually score an overlay tally in THIS run:
            // what the model transports, narrowed by the tally's ParticleType
            // filter. Missing data for a species that will score is fatal --
            // continuing would score the plain, un-responded quantity (or a flat
            // zero) behind a warning (issue #288). Missing data for a species
            // that cannot score is irrelevant and stays a warning.
            let model_has_neutrons = self
                .sources
                .iter()
                .any(|s| s.particle_type() == yamc_particle::particle::ParticleType::Neutron);
            let model_has_photons = self.has_photons();
            for tally_arc in &self.tallies {
                if !tally_arc.multiply_density {
                    let filtered = tally_arc.filters.iter().find_map(|f| match f {
                        yamc_tallies::filter::Filter::ParticleType(pf) => Some(pf.particle_type),
                        _ => None,
                    });
                    let allows =
                        |p: yamc_particle::particle::ParticleType| filtered.is_none_or(|f| f == p);
                    tally_arc.prepare_overlay_xs(
                        &mt_filter,
                        &all_photon_paths,
                        model_has_neutrons
                            && allows(yamc_particle::particle::ParticleType::Neutron),
                        model_has_photons && allows(yamc_particle::particle::ParticleType::Photon),
                    )?;
                }
            }
        }

        // Load D1S chain file and precompute D1S photon data if decay photon mode is enabled.
        // Chain data replaces prompt photon yields with decay photon yields at setup time,
        // avoiding per-collision chain lookups during transport.
        // Registry for interning nuclide names used on the hot path (Particle.parent_nuclide
        // and ParentNuclideFilter). Populated during D1S precompute (which re-Arcs the
        // materials), then used to resolve any parent-nuclide filters on the tallies.
        // Runs before the `tallies`/`base_seed` reads below so its &mut self borrow
        // doesn't overlap the immutable `tallies` borrow (both are otherwise independent).
        let mut nuclide_registry = yamc_nuclide::nuclide_registry::NuclideRegistry::new();
        let decay_photon_nuclide_data =
            self.prepare_decay_photon_data(&mut nuclide_registry, is_root)?;

        // Lift Arc deref out of the per-particle tally loop below.
        let tallies: Vec<&Tally> = self.tallies.iter().map(Arc::as_ref).collect();
        let base_seed = settings.seed;

        // Resolve any parent-nuclide filters against the registry now that D1S target
        // names have been interned. Filter names that aren't D1S targets still get ids
        // (they simply won't match any particle).
        for tally_arc in &self.tallies {
            if let Some(pnf) = tally_arc.get_parent_nuclide_filter() {
                pnf.resolve(&mut nuclide_registry);
            }
        }

        // Global particle ID counter for tracking (atomic for thread safety)
        // Arc so the heartbeat thread can read this live progress counter
        // (workers fetch_add it per particle; the heartbeat only loads it).
        let global_particle_id = Arc::new(AtomicU64::new(0));

        // Collect tracks from all batches
        let all_tracks: Mutex<Vec<TrackStorage>> = Mutex::new(Vec::new());
        // Which histories to capture (only consulted by RealTracker; the
        // NoOpTracker ignores it). `None` means "no explicit selection" -- the
        // tracker type alone decides whether anything is captured.
        let selection = selection.unwrap_or(HistorySelection::All);

        // Lost particle tracking (thread-safe)
        let lost_particle_count = std::sync::atomic::AtomicUsize::new(0);
        let lost_particles_collected: Mutex<Vec<crate::util::lost_particle::LostParticle>> =
            Mutex::new(Vec::new());
        let max_lost = self.max_lost_particles;

        let transport_loop_start = crate::util::timer::Timer::start();

        #[cfg(feature = "debug_diagnostics")]
        photon_diag::reset();

        // Per-tally sizing for the per-history Welford worker scratch.
        // Every tally uses per-history Welford now.
        let any_welford = !tallies.is_empty();
        let welford_tally_num_bins: Vec<usize> = tallies.iter().map(|t| t.num_bins()).collect();

        // Per-history Welford worker state, hoisted out of the per-batch
        // rayon fold so allocation happens once per simulation instead of
        // once per batch. Slot per rayon thread; looked up inside the fold
        // body via `rayon::current_thread_index()`. The mutex is never
        // contended (each thread always locks its own slot); using
        // `parking_lot::Mutex` over `std::sync::Mutex` cuts the
        // uncontended lock+unlock from ~30 ns to ~5 ns.
        let n_rayon_threads = local_pool
            .as_ref()
            .map(|p| p.current_num_threads())
            .unwrap_or_else(rayon::current_num_threads);
        let welford_workers: Vec<parking_lot::Mutex<yamc_tallies::welford::WelfordWorkerState>> =
            (0..n_rayon_threads)
                .map(|_| {
                    parking_lot::Mutex::new(yamc_tallies::welford::WelfordWorkerState::new(
                        &welford_tally_num_bins,
                    ))
                })
                .collect();

        // Per-tally convergence history (aggregate statistics versus number
        // of histories), recorded at each batch checkpoint from the per-worker
        // aggregate accumulators. Installed onto each tally after the loop.
        let mut convergence_history: Vec<Vec<yamc_tallies::ConvergencePoint>> =
            vec![Vec::new(); tallies.len()];

        // Pure Woodcock requires a majorant that bounds Σ_t along the entire
        // flight path (see `build_majorant`); the photon majorant covers the
        // photon channel when photons can appear.
        let majorant = self.build_majorant();
        let photon_majorant = self.build_photon_majorant();

        // Per-cell mean chord lengths for the Woodcock hybrid dispatch
        // (cells-per-mean-free-path decision). Computed once from each
        // cell's region bounding box; an unbounded cell yields infinity,
        // which always surface-tracks it. Only the Hybrid mode consults
        // these; pure Woodcock never falls back, so it leaves this empty.
        let cell_mean_chords: Vec<f64> = if self.tracking_mode == TrackingMode::Hybrid {
            self.geometry
                .cells()
                .iter()
                .map(|c| cell_mean_chord(&c.region.bounding_box()))
                .collect()
        } else {
            Vec::new()
        };

        // Particles actually processed. Equals `total_particles` on a normal
        // run; an early stop (`max_runtime` or `convergence_targets`) finishes
        // after fewer, and the post-loop summary / throughput report this real
        // count rather than the full request. Set from the chunk cursor once
        // the batch loop exits (whether by break or natural completion).
        let particles_run: u64;

        {
            // Progress reporting is in terms of total particles (not
            // batches) so it remains coherent across whatever batch
            // count tune-tallies / the user picked. Print every time
            // we cross a 10%-of-total boundary (so the user sees
            // ~10 progress lines regardless of M).
            // 0 means "unknown total" (uncapped run): the progress renderers
            // then show a running count with no percentage or ETA.
            let total_particles_for_progress = settings.total_particles.unwrap_or(0);
            let mut last_progress_bucket: i64 = -1;

            // In-place "live" status panel: on a TTY we redraw a fixed set
            // of lines (progress/ETA, plus per-tally stats when `tally` is
            // on) using ANSI cursor-up, instead of appending a line per
            // checkpoint. Non-TTY output (pipes, Jupyter, log files, CI)
            // keeps the plain append path so it stays clean and grep-able.
            // Choose the progress sink. A real terminal gets the multi-line
            // ANSI panel; a Jupyter kernel (captured stdout, not a TTY, but
            // renders `\r` in place -- detected via the JPY_PARENT_PID env
            // var the kernel sets) gets a single carriage-return-updated
            // line; everything else (pipes, files, CI) gets plain appended
            // lines. Both panel modes run the heartbeat so progress/ETA stay
            // live between chunk boundaries.
            let is_tty = std::io::stdout().is_terminal();
            let in_jupyter = !is_tty && std::env::var_os("JPY_PARENT_PID").is_some();
            let wants_progress = self.verbose.shows_progress() || self.verbose.tally_stream;
            let live_status = is_root && is_tty && wants_progress;
            let jupyter_status = is_root && in_jupyter && wants_progress;
            let panel_active = live_status || jupyter_status;

            // Heartbeat thread: re-draws progress/ETA periodically even
            // mid-chunk (read from the global particle counter), so a long
            // chunk on a large run never looks hung. Parked between wakeups
            // (~zero CPU), only reads an already-hot atomic, never touches the
            // per-particle path. Non-wasm only (no threads on wasm32).
            let live_panel = Arc::new(Mutex::new(LivePanel::default()));
            let heartbeat_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let heartbeat_handle: Option<std::thread::JoinHandle<()>> = {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if panel_active {
                        const HEARTBEAT_SECS: u64 = 60;
                        let panel = Arc::clone(&live_panel);
                        let counter = Arc::clone(&global_particle_id);
                        let stop = Arc::clone(&heartbeat_stop);
                        // 0 = uncapped (no total to scale against or cap to).
                        let total = settings.total_particles.unwrap_or(0) as u64;
                        let ranks = mpi_size.max(1) as u64;
                        let spawn_elapsed = transport_loop_start.elapsed_secs();
                        let show_progress = self.verbose.progress;
                        let show_eta = self.verbose.eta;
                        let show_tally = self.verbose.tally_stream;
                        let jupyter = jupyter_status;
                        Some(std::thread::spawn(move || {
                            let start = std::time::Instant::now();
                            let step = std::time::Duration::from_millis(200);
                            let interval = std::time::Duration::from_secs(HEARTBEAT_SECS);
                            let mut since = std::time::Duration::ZERO;
                            while !stop.load(std::sync::atomic::Ordering::Acquire) {
                                std::thread::sleep(step);
                                since += step;
                                if since < interval {
                                    continue;
                                }
                                since = std::time::Duration::ZERO;
                                // Root's counter covers only its own MPI
                                // share; scale to a global estimate (ranks==1
                                // for the common single-process case). An
                                // uncapped run (total == 0) has nothing to cap
                                // against, so report the raw scaled count.
                                let scaled = counter
                                    .load(std::sync::atomic::Ordering::Relaxed)
                                    .saturating_mul(ranks);
                                let done = if total == 0 {
                                    scaled
                                } else {
                                    scaled.min(total)
                                };
                                let elapsed = spawn_elapsed + start.elapsed().as_secs_f64();
                                let mut p = panel.lock().unwrap_or_else(|e| e.into_inner());
                                render_live_panel(
                                    &mut p,
                                    done,
                                    total,
                                    elapsed,
                                    show_progress,
                                    show_eta,
                                    show_tally,
                                    jupyter,
                                );
                            }
                        }))
                    } else {
                        None
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    None
                }
            };
            // RAII: stops + joins the heartbeat even on early return / panic.
            let hb_guard = HeartbeatGuard {
                stop: Arc::clone(&heartbeat_stop),
                handle: heartbeat_handle,
            };

            // Read-only inputs shared by every particle history across all
            // batches: gathered once here and threaded into the transport
            // entry points as a single `&TransportCtx` (see
            // `transport::TransportCtx`).
            // Survival-biasing parameters for the transport context: the
            // scalar form the hot loop consumes. Defaults are placeholders
            // when the technique is off (the bool gates all use).
            let survival = self.survival_biasing().copied().unwrap_or_default();
            // Per-tally eligibility for true track-length scoring along
            // Woodcock flight segments (issue #350); surface tracking
            // never consults it, so it stays all-false there.
            let woodcock_tl_mesh_eligible: Vec<bool> =
                if self.tracking_mode == TrackingMode::Surface {
                    vec![false; tallies.len()]
                } else {
                    tallies
                        .iter()
                        .map(|t| crate::transport::woodcock_mesh_track_length_eligible(t))
                        .collect()
                };
            // Weight-window maps borrowed for the run (empty if none). Slice
            // of references, mirroring `tallies`; consumed at collisions.
            let weight_window_refs = self.weight_windows();
            // Flattened per-(nuclide, MT) inelastic kinematics tables, built
            // lazily on first use and shared across threads (issue #111). Its
            // keys are nuclide addresses, so it is deliberately scoped to this
            // run: it is dropped here, well before the nuclide data it was
            // filled from.
            let inelastic_flat_cache = InelasticFlatCache::default();
            let transport_ctx = TransportCtx {
                geometry: &self.geometry,
                photon_cutoff_energy: self.photon_cutoff_energy,
                transport_secondary_photons: self.transport_secondary_photons,
                use_decay_photons: self.use_decay_photons,
                free_gas_threshold: self.free_gas_threshold,
                survival_biasing: self.survival_biasing().is_some(),
                weight_cutoff: survival.weight_cutoff,
                weight_survive: survival.weight_survive,
                tallies: &tallies,
                woodcock_tl_mesh_eligible: &woodcock_tl_mesh_eligible,
                weight_windows: &weight_window_refs,
                transmutation_tallies: transmutation_tallies.as_deref(),
                decay_photon_nuclide_data: &decay_photon_nuclide_data,
                lost_particle_count: &lost_particle_count,
                lost_particles_collected: &lost_particles_collected,
                max_lost,
                debug,
                inelastic_flat_cache: &inelastic_flat_cache,
            };
            let transport_ctx = &transport_ctx;

            // Global index of the first particle in the current chunk. Per-
            // particle seeds are keyed purely to the global particle index
            // (base_seed + global_index * PARTICLE_SEED_STRIDE), so they are
            // independent of how particles are grouped into chunks -- chunk
            // shape never affects results. Advances by each chunk's size
            // (uniform for now; a non-uniform schedule advances by its size).
            let mut particles_before_chunk: u64 = 0;
            // Capped runs iterate `num_chunks` chunks (sizes from
            // `derive_chunk_size`) and stop when exhausted; uncapped runs
            // (`total_particles == None`) draw sizes from `uncapped_chunk_size`
            // and rely on the convergence / `max_runtime` early-stops below to
            // break. `batch` is the running chunk index in both cases.
            let mut batch = 0usize;
            // Built once for the whole run: with a many-source model (a
            // parametric plasma source is one ring source per mesh voxel per
            // reaction) rebuilding the strength table per history would cost
            // more than the transport.
            let source_selector = SourceSelector::new(&self.sources);
            loop {
                let chunk_size = match settings.total_particles {
                    Some(total) => {
                        if batch >= num_chunks {
                            break;
                        }
                        derive_chunk_size(total, num_chunks, batch)
                    }
                    None => uncapped_chunk_size(batch),
                };
                // Per-chunk MPI decomposition, since chunk sizes vary.
                let (local_particles, particle_offset) =
                    mpi_decompose(chunk_size, mpi_rank, mpi_size);
                let chunk_start_global = particles_before_chunk;
                particles_before_chunk = particles_before_chunk.wrapping_add(chunk_size as u64);
                #[cfg(feature = "debug_diagnostics")]
                let batch_start = std::time::Instant::now();
                #[cfg(feature = "debug_diagnostics")]
                let batch_secondary_count = std::sync::atomic::AtomicU64::new(0);
                // Zeroed per batch so the report below counts this batch alone.
                // A static rather than a local because the transport loop
                // increments it from outside this scope.
                #[cfg(feature = "debug_diagnostics")]
                crate::transport::debug::BATCH_COLLISIONS
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                if is_root {
                    let particles_done = chunk_start_global as usize;
                    let bucket = if total_particles_for_progress == 0 {
                        0
                    } else {
                        ((particles_done as u128 * 10) / total_particles_for_progress as u128)
                            as i64
                    };
                    let crossed_boundary = bucket != last_progress_bucket;
                    // In live mode the progress line is redrawn in place at
                    // end-of-batch, so skip the throttled start-of-batch print.
                    if crossed_boundary && !panel_active {
                        last_progress_bucket = bucket;
                        if self.verbose.shows_progress() {
                            if self.verbose.eta {
                                if batch == 0 {
                                    println!(
                                        "Progress: 0/{total} particles (0%)",
                                        total = total_particles_for_progress,
                                    );
                                } else {
                                    // Average wall-clock so far is over the
                                    // completed chunks.
                                    let elapsed = transport_loop_start.elapsed_secs();
                                    let frac_done =
                                        particles_done as f64 / total_particles_for_progress as f64;
                                    let eta = if frac_done > 0.0 {
                                        elapsed * (1.0 - frac_done) / frac_done
                                    } else {
                                        0.0
                                    };
                                    println!(
                                        "Progress: {particles_done}/{total} particles ({pct}%) \
                                         (elapsed {el}, ETA ~{eta_s})",
                                        total = total_particles_for_progress,
                                        pct = bucket * 10,
                                        el = format_secs(elapsed),
                                        eta_s = format_secs(eta),
                                    );
                                }
                            } else {
                                println!(
                                    "Progress: {particles_done}/{total} particles ({pct}%)",
                                    total = total_particles_for_progress,
                                    pct = bucket * 10,
                                );
                            }
                        }
                    }
                }

                // Parallel execution per batch using fold to collect tracker state.
                // Welford worker state is looked up per-thread from the outer-scoped
                // `welford_workers` pool so it persists across batches (avoids the
                // ~28 MB/worker per-batch reallocation cost that originally forced
                // the separate single-fold path).
                let welford_workers_ref = &welford_workers;
                let majorant_ref = majorant.as_deref();
                let photon_majorant_ref = photon_majorant.as_deref();
                let cell_mean_chords_ref = cell_mean_chords.as_slice();
                let tracking_mode = self.tracking_mode;
                let parallel_batch = || {
                    let storage = (0..local_particles)
                        .into_par_iter()
                        .fold(
                            || {
                                // Each thread gets its own state
                                let tracker = T::new_with_selection(selection.clone());
                                let neighbor_lists = NeighborLists::new(self.geometry.num_cells());
                                (
                                    ParticleBank::with_capacity(64),
                                    FastRng::new(0),
                                    tracker,
                                    neighbor_lists,
                                )
                            },
                            |(
                                mut particle_bank,
                                mut thread_rng,
                                mut thread_tracker,
                                mut thread_neighbors,
                            ),
                             local_particle_idx| {
                                let rng = &mut thread_rng;
                                let tracker = &mut thread_tracker;
                                let neighbor_lists = &mut thread_neighbors;
                                // Clear the bank for reuse (preserves capacity, avoids reallocation)
                                particle_bank.clear();

                                // Map local particle index to global particle index
                                let particle_idx = particle_offset + local_particle_idx;

                                // Each particle gets a unique, reproducible seed.
                                // The stride is shared with the combine_results
                                // stream-overlap validation so they cannot drift.
                                const PARTICLE_STRIDE: u64 =
                                    yamc_tallies::combine::PARTICLE_SEED_STRIDE;
                                // Seed = base_seed + global_particle_index * STRIDE,
                                // where global index = chunk start + index within
                                // chunk. No batch term: the seed depends only on the
                                // particle's global identity, not the chunk grouping.
                                let global_particle_index =
                                    chunk_start_global.wrapping_add(particle_idx as u64);
                                let particle_seed = base_seed.wrapping_add(
                                    global_particle_index.wrapping_mul(PARTICLE_STRIDE),
                                );

                                // Reseed FastRng for this particle (avoids struct allocation)
                                rng.reseed(particle_seed);

                                // Root of this history's collision-stream tree
                                // (issues #111, #274, #315). Built by the shared
                                // `history_seed` -- the single definition of
                                // per-history seeding, also used to build the
                                // GPU seed buffer -- so the CPU and GPU consume
                                // the identical 64-bit stream and per-history
                                // diffing (#40) is bit-exact. Keyed on the base
                                // seed as well as the global index (#315), so
                                // re-running with a different `seed` gives an
                                // independent collision realisation, not just a
                                // re-sampled source. The source particle
                                // transports on it; each banked secondary gets
                                // its OWN seed derived from it, and the pop loop
                                // below expands whichever seed it pops.
                                let source_seed = yamc_rng::history_seed(
                                    base_seed,
                                    global_particle_index,
                                );
                                // Declared here, assigned at every pop below.
                                let mut pcg_state: u64;

                                // Start tracking this history, keyed by its
                                // global index so the captured set is the same
                                // regardless of chunking/thread scheduling.
                                tracker.start_history(batch, global_particle_index as usize);

                                // Get unique particle ID for tracking
                                let source_particle_id = global_particle_id
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                                // Start with the source particle
                                let mut particle =
                                    self.sample_source_with(&source_selector, rng);
                                particle.alive = true;
                                particle_bank.add_source_particle(particle.clone(), source_seed);

                                // Find initial cell for source birth tracking
                                let initial_cell_id = self
                                    .geometry
                                    .find_cell_index((
                                        particle.position[0],
                                        particle.position[1],
                                        particle.position[2],
                                    ))
                                    .and_then(|idx| self.geometry.cells()[idx].cell_id);

                                // Record source birth
                                tracker.record_birth(
                                    source_particle_id,
                                    None, // No parent for source particles
                                    0,    // Generation 0
                                    particle.position,
                                    particle.direction,
                                    particle.energy,
                                    particle.weight,
                                    initial_cell_id,
                                    "source",
                                );

                                // Track particle metadata for genealogy
                                let current_particle_id = source_particle_id;
                                let current_parent_id: Option<u64> = None;
                                let current_generation: u32 = 0;

                                // Lock this thread's welford worker slot for the
                                // duration of this source history. No contention --
                                // each rayon thread always touches the same slot.
                                let thread_idx = rayon::current_thread_index().unwrap_or(0);
                                let mut welford_worker = welford_workers_ref[thread_idx].lock();

                                // Process all particles in the bank (primary + secondaries)
                                let mut particles_processed = 0;
                                // Last-resort ceiling on the per-history population.
                                // Weight-window splitting is bounded by a per-history
                                // split budget (WW_SPLIT_BUDGET_PER_HISTORY in transport),
                                // which stops splitting while leaving particles intact
                                // (weight conserved), so the bank drains well below this
                                // ceiling. Reaching it means the history is not draining
                                // at all, which in practice means a multiplying (near- or
                                // supercritical) chain, so it ABORTS rather than breaking
                                // out (issue #348): truncating here would discard every
                                // still-queued particle's weight and bias every tally low,
                                // silently. The fission-progeny path has its own, earlier
                                // and better-diagnosed ceiling
                                // (FISSION_BANK_LIMIT_PER_HISTORY in transport::fission);
                                // this one is the backstop for a chain that grows too
                                // slowly to trip it.
                                const MAX_PARTICLES_PER_HISTORY: usize = 50_000;

                                while let Some(mut particle) = particle_bank.pop_particle() {
                                    // Start this walk on ITS OWN stream (issue
                                    // #111). The seed came from the particle's
                                    // place in the history's emission tree, not
                                    // from where the parent's state happened to
                                    // be when the walk was scheduled, so the
                                    // drain order no longer selects the physics
                                    // and the GPU's in-thread FIFO reproduces
                                    // this LIFO stack's results. For the source
                                    // particle (always the first pop) this is
                                    // `source_seed`, so a history with no
                                    // secondaries is byte-identical to before.
                                    pcg_state = yamc_rng::expand_seed(
                                        particle_bank.walk_seed(),
                                    );
                                    particles_processed += 1;
                                    #[cfg(feature = "debug_diagnostics")]
                                    batch_secondary_count
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    if particles_processed > MAX_PARTICLES_PER_HISTORY {
                                        let queued = particle_bank.len();
                                        let progeny = particle_bank.fission_progeny();
                                        // Name the cause only when fission actually
                                        // accounts for the population. A trace-fissile
                                        // model that fails to drain for some other reason
                                        // gets the bare count instead of a wrong diagnosis.
                                        let diagnosis = if progeny * 4 >= particles_processed {
                                            format!(
                                                " {progeny} fission neutrons were produced \
                                                 along the way, so this geometry is at or \
                                                 above critical; yamc does fixed-source \
                                                 transport only and has no \
                                                 criticality/eigenvalue mode."
                                            )
                                        } else if progeny > 0 {
                                            format!(" {progeny} of them were fission neutrons.")
                                        } else {
                                            String::new()
                                        };
                                        panic!(
                                            "Source history {global_particle_index} did not \
                                             drain: {MAX_PARTICLES_PER_HISTORY} particles \
                                             transported with {queued} still \
                                             queued.{diagnosis} The run stops here rather \
                                             than truncating the history, because dropping \
                                             the queued particles would destroy their \
                                             weight and bias every tally low without saying \
                                             so."
                                        );
                                    }

                                    match tracking_mode {
                                        TrackingMode::Surface => {
                                            transport_particle(
                                                transport_ctx,
                                                &mut particle,
                                                rng,
                                                &mut pcg_state,
                                                tracker,
                                                neighbor_lists,
                                                &mut particle_bank,
                                                current_particle_id,
                                                current_parent_id,
                                                current_generation,
                                                particle_idx,
                                                &mut welford_worker,
                                            );
                                        }
                                        TrackingMode::Woodcock | TrackingMode::Hybrid => {
                                            // Validated at simulate_transport
                                            // start: majorant is Some here.
                                            // `photon_majorant_ref` is Some
                                            // whenever transport_secondary_photons is on.
                                            // `cell_mean_chords_ref` is only
                                            // read when `hybrid` is true.
                                            transport_particle_woodcock(
                                                transport_ctx,
                                                &mut particle,
                                                majorant_ref.expect(
                                                    "Woodcock majorant must be built for tracking_mode=Woodcock/Hybrid",
                                                ),
                                                photon_majorant_ref,
                                                cell_mean_chords_ref,
                                                tracking_mode == TrackingMode::Hybrid,
                                                rng,
                                                &mut pcg_state,
                                                tracker,
                                                neighbor_lists,
                                                &mut particle_bank,
                                                current_particle_id,
                                                current_parent_id,
                                                current_generation,
                                                particle_idx,
                                                &mut welford_worker,
                                            );
                                        }
                                    }
                                } // end of particle queue processing while loop

                                // Finish tracking this history
                                tracker.finish_history();
                                // No-op when no tally uses per-history Welford
                                // (zero-sized scratch ⇒ touched_bins empty).
                                welford_worker.finish_history();
                                // Lock automatically released at end of scope.

                                // Return state for the next iteration
                                (particle_bank, thread_rng, thread_tracker, thread_neighbors)
                            },
                        )
                        .map(|(_, _, tracker, _)| tracker.into_storage())
                        .reduce(TrackStorage::new, |mut a, b| {
                            a.tracks.extend(b.tracks);
                            a
                        });
                    storage
                };

                // Execute the parallel batch either within the local pool (if provided) or default pool
                let batch_storage = if let Some(pool) = &local_pool {
                    pool.install(parallel_batch)
                } else {
                    parallel_batch()
                };

                // Collect tracking results from this batch
                all_tracks
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(batch_storage);

                // Print per-batch diagnostics (only with debug_diagnostics feature)
                #[cfg(feature = "debug_diagnostics")]
                if is_root {
                    let batch_secs = batch_start.elapsed().as_secs_f64();
                    let n_secondary =
                        batch_secondary_count.load(std::sync::atomic::Ordering::Relaxed);
                    let n_collisions = crate::transport::debug::BATCH_COLLISIONS
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let per_source = n_secondary as f64 / local_particles as f64;
                    let decay_photons =
                        photon_diag::DECAY_PHOTONS.load(std::sync::atomic::Ordering::Relaxed);
                    let fluor =
                        photon_diag::FLUORESCENCE_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
                    let ttb = photon_diag::TTB_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
                    let pe = photon_diag::PE_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
                    println!(
                    "  Batch {} done in {:.3}s | particles: {} ({:.1}/source) | collisions: {} | D1S: {} | fluor: {} | TTB: {} | PE: {}",
                    batch + 1, batch_secs, n_secondary, per_source, n_collisions, decay_photons, fluor, ttb, pe
                );
                }

                // Accumulate batch results into running totals for online variance estimation
                for tally in tallies.iter() {
                    tally.accumulate_batch(chunk_size as u32);
                }
                if let Some(dep_tallies) = &transmutation_tallies {
                    // Pass the real per-chunk source-particle count so the tally
                    // normalizes by the true total, not the CPU chunk count
                    // (issue #128). Mirrors the main tally above.
                    dep_tallies.accumulate_batch(chunk_size);
                }

                // --- Convergence snapshot + precision targets ---
                // Combine the per-worker aggregate moments / PDF (a handful of
                // scalars + a small histogram each -- off the per-event hot
                // path) into one point per tally, so users can watch
                // convergence and optionally stop the run early. Workers are
                // idle between batches, so these locks are uncontended.
                // Per-tally stream lines for this batch, collected for the
                // in-place live render below (printed directly in non-live mode).
                let mut stream_lines: Vec<String> = Vec::new();
                if any_welford {
                    let elapsed = transport_loop_start.elapsed_secs();
                    // Per-tally aggregate moments for THIS rank. Under MPI they are
                    // reduced below before any convergence target is tested, so the
                    // decision is made on the true global statistics rather than a
                    // rank's own share (issue #241).
                    let mut local_aggs: Vec<yamc_tallies::welford::AggMoments> =
                        Vec::with_capacity(tallies.len());
                    for (ti, tally) in tallies.iter().enumerate() {
                        let mut agg = yamc_tallies::welford::AggMoments::ZERO;
                        let mut pdf = yamc_tallies::welford::ScorePdf::default();
                        for w in welford_workers.iter() {
                            let g = w.lock();
                            if let Some(tw) = g.tallies.get(ti) {
                                agg.combine(&tw.agg);
                                pdf.combine(&tw.score_pdf);
                            }
                        }
                        let std = if agg.n >= 2 {
                            let n = agg.n as f64;
                            (agg.m2 / ((n - 1.0) * n)).max(0.0).sqrt()
                        } else {
                            0.0
                        };
                        let rel = if agg.mean.abs() > 0.0 {
                            std / agg.mean.abs()
                        } else {
                            0.0
                        };
                        let vov = agg.variance_of_variance();
                        local_aggs.push(agg);
                        let fom = if rel > 0.0 && elapsed > 0.0 {
                            1.0 / (rel * rel * elapsed)
                        } else {
                            0.0
                        };
                        convergence_history[ti].push(yamc_tallies::ConvergencePoint {
                            n_histories: agg.n,
                            mean: agg.mean,
                            relative_error: rel,
                            variance_of_variance: vov,
                            figure_of_merit: fom,
                            tail_slope: pdf.tail_slope(),
                        });
                        // Live convergence line from the freshly-combined
                        // aggregate. Mesh tallies are skipped: their summed-
                        // across-bins mean isn't physically meaningful (users
                        // care about per-voxel error, not a single scalar).
                        if is_root && self.verbose.tally_stream && !tally.has_mesh_filter() {
                            let name = tally.name.as_deref().unwrap_or("(unnamed)");
                            let line = if rel > 0.0 {
                                format!(
                                    "  {name:<24} {:.4e} ± {std:.2e}   ({:.1}% r.e.)",
                                    agg.mean,
                                    rel * 100.0
                                )
                            } else {
                                format!("  {name:<24} {:.4e}", agg.mean)
                            };
                            if panel_active {
                                stream_lines.push(line);
                            } else {
                                println!("{line}");
                            }
                        }
                    }

                    // --- Convergence targets, collectively under MPI ---
                    // Each rank transports its own share, so its aggregate is a
                    // partial view: testing a target against it would stop the run
                    // at the wrong precision, and ranks disagreeing about when to
                    // stop would desync the post-loop gather collectives. So the
                    // per-tally moments are gathered to root, folded there with
                    // Chen's exact parallel combine (the same fold the post-loop
                    // tally reduction uses), evaluated once, and the single decision
                    // bit is broadcast -- the same shape as the `max_runtime`
                    // collective stop. Evaluating on root and broadcasting, rather
                    // than each rank allreducing and deciding for itself, keeps the
                    // decision bit-identical across ranks: a borderline target must
                    // not flip on summation order.
                    //
                    // The collective runs at every checkpoint whenever targets are
                    // set (and not at all when they are not), so every rank reaches
                    // it in lockstep.
                    let all_targets_met = if self.convergence_targets.is_empty() {
                        false
                    } else if mpi_size > 1 {
                        let mut packed: Vec<f64> = Vec::with_capacity(local_aggs.len() * 5);
                        for a in &local_aggs {
                            packed.extend_from_slice(&[a.n as f64, a.mean, a.m2, a.m3, a.m4]);
                        }
                        let gathered = mpi_ctx.gather_f64(&packed, 0);
                        let mut decision = [0.0_f64];
                        if let Some(g) = gathered {
                            let n_tallies = local_aggs.len();
                            let mut folded =
                                vec![yamc_tallies::welford::AggMoments::ZERO; n_tallies];
                            for r in 0..mpi_size as usize {
                                for (t, slot) in folded.iter_mut().enumerate() {
                                    let off = (r * n_tallies + t) * 5;
                                    slot.combine(&yamc_tallies::welford::AggMoments {
                                        n: g[off] as u64,
                                        mean: g[off + 1],
                                        m2: g[off + 2],
                                        m3: g[off + 3],
                                        m4: g[off + 4],
                                    });
                                }
                            }
                            decision[0] = if convergence_targets_met(
                                &folded,
                                &tallies,
                                &self.convergence_targets,
                            ) {
                                1.0
                            } else {
                                0.0
                            };
                        }
                        mpi_ctx.broadcast_f64(&mut decision, 0);
                        decision[0] > 0.5
                    } else {
                        convergence_targets_met(&local_aggs, &tallies, &self.convergence_targets)
                    };
                    if all_targets_met {
                        if is_root && self.verbose.summary {
                            let done = particles_before_chunk as usize;
                            println!(
                                "Convergence targets satisfied after {done} particles; \
                                 stopping early."
                            );
                        }
                        break;
                    }
                }

                // --- Wall-time budget (max_runtime) early stop ---
                // Stop at the first chunk checkpoint where elapsed wall time has
                // reached the budget. The current chunk is already accumulated,
                // so post-loop finalization yields valid statistics for the
                // histories completed. Under MPI the decision is COLLECTIVE:
                // every rank checks its own elapsed time, then `any_rank_true`
                // OR-reduces across ranks (any rank over budget => all stop) and
                // broadcasts the one stop bit, so ranks break at the SAME
                // checkpoint and never desync the post-loop gather collectives.
                // `any_rank_true` returns the local value for mpi_size == 1, so
                // single-process behaviour is unchanged. Every rank must reach
                // the collective in lockstep, so it runs at every checkpoint
                // when a budget is set (and is skipped entirely when it isn't).
                if settings.max_runtime.is_some() {
                    let local_over = runtime_budget_exhausted(
                        settings.max_runtime,
                        transport_loop_start.elapsed_secs(),
                    );
                    if mpi_ctx.any_rank_true(local_over) {
                        if is_root && self.verbose.summary {
                            let done = particles_before_chunk as usize;
                            let budget = settings.max_runtime.unwrap_or(0.0);
                            println!(
                                "Max runtime ({budget:.3}s) reached after {done} particles; \
                                 stopping early."
                            );
                        }
                        break;
                    }
                }

                // Live in-place render at this chunk boundary. Refresh the
                // tally snapshot (the heartbeat shows its age) and redraw
                // progress/ETA + tally under the panel lock, so the boundary
                // render and the heartbeat never fight over the cursor.
                // `done` comes from the global counter for consistency with
                // the heartbeat's mid-chunk renders.
                if panel_active {
                    // 0 = uncapped: the renderer shows a count with no %/ETA.
                    let total = settings.total_particles.unwrap_or(0) as u64;
                    let elapsed = transport_loop_start.elapsed_secs();
                    // Exact global particles done through this chunk (the
                    // per-rank counter would undercount under MPI).
                    let done = if total == 0 {
                        particles_before_chunk
                    } else {
                        particles_before_chunk.min(total)
                    };
                    let mut p = live_panel.lock().unwrap_or_else(|e| e.into_inner());
                    if self.verbose.tally_stream {
                        p.tally_values = std::mem::take(&mut stream_lines);
                        p.checkpoint_elapsed_secs = elapsed;
                    }
                    render_live_panel(
                        &mut p,
                        done,
                        total,
                        elapsed,
                        self.verbose.progress,
                        self.verbose.eta,
                        self.verbose.tally_stream,
                        jupyter_status,
                    );
                } else if is_root && !self.verbose.is_silent() {
                    // Plain non-TTY (pipe / CI / file): flush so the appended
                    // progress + tally lines stream out progressively instead
                    // of being block-buffered to the end of the run.
                    let _ = std::io::stdout().flush();
                }
                batch += 1;
            } // end of batch loop
            drop(hb_guard); // stop + join the heartbeat before the summary

            // The chunk cursor now holds the cumulative count through the last
            // fully-accumulated chunk: equal to `total_particles` on a normal
            // run, or the partial count when an early stop broke the loop.
            particles_run = particles_before_chunk;

            // Finalize the live panel: Jupyter's single `\r` line needs a
            // trailing newline so the summary starts on its own line; the
            // plain non-TTY path didn't print a completion line during the
            // loop (its progress lines are start-of-chunk), so emit one here.
            // Report the real count and percentage, which is below 100% after
            // an early stop.
            if jupyter_status {
                println!();
            } else if is_root && !panel_active && self.verbose.shows_progress() && num_chunks > 0 {
                // `num_chunks > 0` only for a capped run, so the total is Some.
                let total = settings.total_particles.unwrap_or(0);
                let done = (particles_run as usize).min(total);
                let pct = done.saturating_mul(100).checked_div(total).unwrap_or(100);
                println!("Progress: {done}/{total} particles ({pct}%)");
                let _ = std::io::stdout().flush();
            }
        }

        // Finalize the per-history Welford accumulator only if any tally
        // actually requested WelfordPerHistory. Combine all per-thread
        // worker states, run the lazy zero-fold, and install per-tally
        // global stats onto each WelfordPerHistory tally so its
        // `get_mean` / `get_std_dev` / `total_mean` / `total_std` route
        // through the per-history Welford state. Tallies using BatchedSumSq
        // or Welford (Stage 1) are left alone -- their sum / sum_sq /
        // mean / m2 have already been populated by accumulate_batch in
        // the per-batch loop above.
        if any_welford {
            let combined: Option<yamc_tallies::welford::WelfordWorkerState> = welford_workers
                .into_iter()
                .map(|m| m.into_inner())
                .reduce(|a, b| a.combine(b));
            if let Some(combined) = combined {
                let mut global_stats = combined.finalize();
                // Under MPI each rank's Welford state covers only its own
                // particle range. Gather the per-rank raw (mean, m2, n)
                // state to root and fold in rank order with the same Chen
                // combine used for thread reduction, so rank 0 installs
                // the complete statistics. (A moment-space reduce_sum
                // would be cheaper but reconstructing m2 from summed
                // squares is catastrophically cancellation-prone, so the
                // raw state is gathered instead.) Non-root ranks keep
                // their local partial state; only root results are
                // complete, and `combine_results` refuses non-root MPI
                // results via the run provenance.
                if mpi_size > 1 {
                    use yamc_tallies::welford::WelfordTallyStats;
                    let ranks = mpi_size as usize;
                    let n_local = global_stats
                        .per_tally
                        .first()
                        .map(|t| t.n_histories)
                        .unwrap_or(0);
                    let rank_n = mpi_ctx.gather_u64(&[n_local], 0);
                    for stats in global_stats.per_tally.iter_mut() {
                        let bins = stats.mean.len();
                        if bins == 0 {
                            continue;
                        }
                        let means = mpi_ctx.gather_f64(&stats.mean, 0);
                        let m2s = mpi_ctx.gather_f64(&stats.m2, 0);
                        if let (Some(means), Some(m2s), Some(rank_n)) =
                            (means, m2s, rank_n.as_ref())
                        {
                            let mut folded: Option<WelfordTallyStats> = None;
                            for r in 0..ranks {
                                let rank_stats = WelfordTallyStats {
                                    mean: means[r * bins..(r + 1) * bins].to_vec(),
                                    m2: m2s[r * bins..(r + 1) * bins].to_vec(),
                                    n_histories: rank_n[r],
                                    // The per-history aggregate moments / PDF
                                    // are not gathered across ranks yet, so the
                                    // tally-level reliability stats are reported
                                    // only for single-process runs. (follow-up)
                                    agg: yamc_tallies::welford::AggMoments::ZERO,
                                    score_pdf: yamc_tallies::welford::ScorePdf::default(),
                                };
                                match &mut folded {
                                    None => folded = Some(rank_stats),
                                    Some(acc) => acc
                                        .combine(&rank_stats)
                                        .expect("rank Welford states share bin counts"),
                                }
                            }
                            if let Some(folded) = folded {
                                *stats = folded;
                            }
                        }
                    }
                }
                for (i, tally) in tallies.iter().enumerate() {
                    if let Some(ts) = global_stats.per_tally.get(i) {
                        tally.install_finalized(ts.clone());
                    }
                    tally.install_convergence_history(std::mem::take(&mut convergence_history[i]));
                }
            }
        }

        // Transmutation accumulators are rank-local while their normalisation
        // denominator is global, so they must be summed across ranks before any
        // rate is read (issue #287); without this every inventory came out low
        // by exactly 1/n_ranks. Runs before the barrier below since it is itself
        // collective.
        if let Some(dep_tallies) = &transmutation_tallies {
            dep_tallies.reduce_across_ranks(&mpi_ctx);
        }

        // Final MPI barrier to ensure all ranks finish
        if mpi_size > 1 {
            mpi_ctx.barrier();
        }

        #[cfg(feature = "debug_timing")]
        eprintln!(
            "[TIMING] run_internal particle transport: {:.3}s",
            transport_loop_start.elapsed_secs()
        );

        let transport_secs = transport_loop_start.elapsed_secs();
        let elapsed_secs = start_time.elapsed_secs();
        // Particles actually processed (less than `settings.total_particles` when an
        // early stop fired, and simply the whole count for an uncapped run).
        // Reporting the real count keeps the summary total and the throughput /
        // `last_particles_per_second` accurate, which the figure-of-merit
        // comparison use case relies on.
        let total_particles = settings
            .total_particles
            .map_or(particles_run as usize, |cap| {
                (particles_run as usize).min(cap)
            });
        let particles_per_second = (total_particles as f64 / elapsed_secs) as u64;

        // Store timing results for programmatic access
        self.last_elapsed_secs = Some(elapsed_secs);
        self.last_particles_per_second = Some(particles_per_second);
        self.last_data_load_secs = Some(data_load_secs);
        self.last_transport_secs = Some(transport_secs);

        // Only root rank prints the summary (or all ranks if single process),
        // and only when the `summary` verbose flag is set.
        if is_root && self.verbose.summary {
            self.print_run_summary(
                elapsed_secs,
                data_load_secs,
                transport_secs,
                total_particles,
                particles_per_second,
            );
        }

        // Collect lost particles into the model
        let total_lost = lost_particle_count.load(std::sync::atomic::Ordering::Relaxed);
        self.lost_particles = lost_particles_collected.into_inner().unwrap();
        if total_lost > 0 && is_root {
            eprintln!(
                "WARNING: {} lost particle(s) detected. Geometry may have gaps.",
                total_lost
            );
        }

        // Merge all collected tracks from all batches
        let all_storages = all_tracks.into_inner().unwrap();
        let mut merged = TrackStorage::new();
        for storage in all_storages {
            merged.tracks.extend(storage.tracks);
        }

        // Print tracking summary if enabled
        if print_tracking_summary && is_root && !merged.tracks.is_empty() {
            println!(
                "Tracked {} histories with {} total events",
                merged.tracks.len(),
                merged.total_events()
            );
        }

        Ok(merged)
    }

    /// Print the end-of-run summary on the root rank: timing / throughput
    /// line, per-tally figure-of-merit, and (under the respective debug
    /// features) photon diagnostics and runtime collision counters.
    fn print_run_summary(
        &self,
        elapsed_secs: f64,
        data_load_secs: f64,
        transport_secs: f64,
        total_particles: usize,
        particles_per_second: u64,
    ) {
        // Format numbers with thousand separators
        let format_with_commas = |n: u64| -> String {
            n.to_string()
                .as_bytes()
                .rchunks(3)
                .rev()
                .map(std::str::from_utf8)
                .collect::<Result<Vec<&str>, _>>()
                .unwrap()
                .join(",")
        };

        let formatted_pps = format_with_commas(particles_per_second);
        let formatted_total = format_with_commas(total_particles as u64);

        let transport_pps = if transport_secs > 0.0 {
            format_with_commas((total_particles as f64 / transport_secs) as u64)
        } else {
            "N/A".to_string()
        };

        println!("Transport simulation complete.");
        println!(
            "Time: {elapsed_secs:.3} s (data load: {data_load_secs:.3} s, transport: {transport_secs:.3} s) | \
                 Particles/s: {formatted_pps} (transport only: {transport_pps}) | Total: {formatted_total} particles"
        );

        // Final figure-of-merit per tally -- FOM = 1 / (rel_err² × elapsed),
        // the standard Monte Carlo convergence metric. Computed once
        // against the converged state at end-of-run. Mesh / multi-bin
        // tallies also get min/median/max across nonzero per-bin FOMs so
        // the aggregate doesn't hide bin-level spread on dense meshes.
        if !self.tallies.is_empty() {
            println!("Tally statistics:");
            for tally in &self.tallies {
                let name = tally.name.as_deref().unwrap_or("(unnamed)");
                let result = tally.finalize().with_fom(elapsed_secs);
                let agg = result.aggregate_figure_of_merit;
                let checks = result.statistical_checks();
                let verdict = if checks.n_evaluated() == 0 {
                    "n/a".to_string()
                } else {
                    format!(
                        "{} ({}/{})",
                        if checks.passed() { "PASSED" } else { "FAILED" },
                        checks.n_passed(),
                        checks.n_evaluated()
                    )
                };
                if tally.num_bins() > 1 {
                    let mut nonzero: Vec<f64> = result
                        .figure_of_merit
                        .iter()
                        .copied()
                        .filter(|f| f.is_finite() && *f > 0.0)
                        .collect();
                    if nonzero.is_empty() {
                        println!("  {name}: FOM = {agg:.3e} (no nonzero bins)  checks={verdict}");
                    } else {
                        nonzero.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let min = nonzero[0];
                        let max = nonzero[nonzero.len() - 1];
                        let median = nonzero[nonzero.len() / 2];
                        println!(
                            "  {name}: FOM = {agg:.3e} (per-bin: min={min:.2e}, median={median:.2e}, max={max:.2e})  checks={verdict}",
                        );
                    }
                } else {
                    let mean = result.aggregate_mean();
                    let std = result.aggregate_std_dev();
                    let rel = result.aggregate_relative_error() * 100.0;
                    println!(
                        "  {name}: mean={mean:.4e} +/- {std:.2e} ({rel:.1}% r.e.)  FOM={agg:.3e}  VOV={:.2e}  checks={verdict}",
                        result.aggregate_variance_of_variance(),
                    );
                }
            }
        }

        #[cfg(feature = "debug_diagnostics")]
        photon_diag::print_summary();

        #[cfg(feature = "debug_runtime")]
        {
            let (collisions, scattering, absorption, fission, mt2, mt_other) = get_debug_counters();
            let elastic_count = get_elastic_scatter_count();
            println!(
                "Collisions: {collisions} | Scattering: {scattering} | Absorption: {absorption} | Fission: {fission}"
            );
            println!(
                "Scattering MT breakdown: MT2(elastic)={mt2} | Other={mt_other} | Logged elastic={elastic_count}"
            );
        }
    }

    /// Find all cells with transmutable materials, grouped by material_id.
    ///
    /// Returns: material_id -> Vec<cell_index>
    pub(crate) fn find_transmutable_cells(
        &self,
    ) -> Result<HashMap<u32, Vec<usize>>, Box<dyn std::error::Error>> {
        let mut transmutable: HashMap<u32, Vec<usize>> = HashMap::new();

        for (idx, cell) in self.geometry.cells().iter().enumerate() {
            if let Some(material_arc) = self.geometry.material_for(cell) {
                let material = material_arc.as_ref();
                if material.transmutable {
                    material.validate_for_transmutation().map_err(|e| {
                        format!(
                            "Material {:?} in cell {:?} failed validation: {}",
                            material.material_id, cell.cell_id, e
                        )
                    })?;
                    let mat_id = material.material_id.ok_or_else(|| {
                        format!(
                            "Transmutable material in cell {:?} has no material_id set",
                            cell.cell_id
                        )
                    })?;
                    transmutable.entry(mat_id).or_default().push(idx);
                }
            }
        }

        Ok(transmutable)
    }
}

/// Distribute `particles_per_chunk` particles across MPI ranks
/// (embarrassingly parallel decomposition over the per-chunk slice --
/// each rank processes its share of `particles_per_chunk` particles
/// each batch). Returns `(local_particles, particle_offset)` for the
/// given rank.
/// Whether every active convergence target is satisfied by these per-tally
/// aggregate moments.
///
/// `aggs` is indexed like `tallies`. A target with no tally selector applies to
/// every tally; one with a selector applies only to the tallies it names. All
/// active targets must hold, so the run stops at the first checkpoint where the
/// whole set is satisfied.
///
/// Shared by the single-process path and the MPI path (which passes the
/// rank-folded moments), so both decide from one implementation and cannot
/// drift on the metric formulas.
fn convergence_targets_met(
    aggs: &[yamc_tallies::welford::AggMoments],
    tallies: &[&yamc_tallies::tally::Tally],
    targets: &[yamc_tallies::ConvergenceTarget],
) -> bool {
    if targets.is_empty() {
        return false;
    }
    for (agg, tally) in aggs.iter().zip(tallies.iter()) {
        let std = if agg.n >= 2 {
            let n = agg.n as f64;
            (agg.m2 / ((n - 1.0) * n)).max(0.0).sqrt()
        } else {
            0.0
        };
        let rel = if agg.mean.abs() > 0.0 {
            std / agg.mean.abs()
        } else {
            0.0
        };
        let vov = agg.variance_of_variance();
        for target in targets {
            if !target.targets(tally) {
                continue;
            }
            let value = match target.metric {
                yamc_tallies::ConvergenceMetric::RelativeError => rel,
                yamc_tallies::ConvergenceMetric::StandardDeviation => std,
                yamc_tallies::ConvergenceMetric::VarianceOfVariance => vov,
            };
            if !target.is_satisfied_by(value) {
                return false;
            }
        }
    }
    true
}

fn mpi_decompose(particles_per_chunk: usize, mpi_rank: i32, mpi_size: i32) -> (usize, usize) {
    let particles_per_rank = particles_per_chunk / (mpi_size as usize);
    let extra_particles = particles_per_chunk % (mpi_size as usize);

    // Ranks 0..extra_particles get one extra particle
    let local_particles = if (mpi_rank as usize) < extra_particles {
        particles_per_rank + 1
    } else {
        particles_per_rank
    };

    // Calculate starting particle index for this rank
    let particle_offset = if (mpi_rank as usize) < extra_particles {
        (mpi_rank as usize) * (particles_per_rank + 1)
    } else {
        extra_particles * (particles_per_rank + 1)
            + ((mpi_rank as usize) - extra_particles) * particles_per_rank
    };

    (local_particles, particle_offset)
}

/// Whether a wall-time budget has been exhausted. Returns true once
/// `elapsed_secs` has reached `max_runtime` (when a budget is set); always
/// false when `max_runtime` is `None`. Factored out so the `max_runtime`
/// stop condition is unit-testable without a full transport run.
pub(crate) fn runtime_budget_exhausted(max_runtime: Option<f64>, elapsed_secs: f64) -> bool {
    matches!(max_runtime, Some(budget) if elapsed_secs >= budget)
}

/// Largest number of source particles processed in one CPU transport chunk.
///
/// The loop consults `max_runtime` and refreshes the per-tally convergence
/// snapshot only at chunk boundaries, so bounding the chunk size bounds how far
/// past a wall-clock budget a run can overshoot, and keeps a huge
/// `total_particles` from being scheduled as a few enormous chunks. Issue #193:
/// a `1e12` total made each of the old 10 chunks `1e11`, whose per-chunk
/// allocation failed outright, and `max_runtime` was only consulted between
/// those never-finishing chunks so it never took effect. Mirrors the GPU path's
/// `MAX_PARTICLES_PER_GPU_DISPATCH`, without the watchdog motivation.
///
/// Sized to keep the per-chunk re-init cost negligible on realistic runs while
/// still checking `max_runtime` often. It bounds the overshoot to one chunk's
/// wall-time; that is sub-second for analog transport but can be tens of
/// seconds for weight-window runs, where a single source history processes
/// thousands of split particles (so a timed weight-window run may overrun its
/// budget by up to one chunk).
pub(crate) const MAX_PARTICLES_PER_CPU_CHUNK: usize = 20_000;

/// Number of chunks the CPU transport loop splits `total_particles` into.
///
/// At least `TARGET_CHUNKS` (so short runs still refresh the convergence
/// snapshot ~10 times), but enough that no chunk exceeds
/// [`MAX_PARTICLES_PER_CPU_CHUNK`] -- so `max_runtime` checkpoints frequently
/// and an arbitrarily large `total_particles` yields many small chunks rather
/// than a few enormous ones. Never more chunks than particles.
///
/// Derived from `total_particles` alone, so every MPI rank computes the
/// identical count (required for lock-step collectives); combined with
/// per-history Welford and index-keyed seeds the chunk shape never changes
/// results. Chunk sizes come from [`derive_chunk_size`], which the loop
/// evaluates lazily so a huge total never materialises a giant schedule.
pub(crate) fn derive_chunk_count(total_particles: usize) -> usize {
    if total_particles == 0 {
        return 0;
    }
    const TARGET_CHUNKS: usize = 10;
    total_particles
        .div_ceil(MAX_PARTICLES_PER_CPU_CHUNK)
        .max(TARGET_CHUNKS)
        .min(total_particles)
}

/// Size of chunk `batch` (of `n_chunks`, from [`derive_chunk_count`]) when
/// `total_particles` is split as evenly as possible: the first
/// `total_particles % n_chunks` chunks get one extra particle, so the sizes sum
/// to exactly `total_particles` (max - min <= 1).
pub(crate) fn derive_chunk_size(total_particles: usize, n_chunks: usize, batch: usize) -> usize {
    let base = total_particles / n_chunks;
    let rem = total_particles % n_chunks;
    if batch < rem {
        base + 1
    } else {
        base
    }
}

/// Chunk size for the `batch`-th chunk of an uncapped run
/// (`total_particles == None`), which has no finite schedule. Front-loaded
/// (100, 1_000, 10_000) so the first convergence / `max_runtime` checkpoints
/// arrive almost immediately, then a steady 100_000 for throughput. The run
/// stops only when `max_runtime` or a convergence target trips, so this size
/// just sets the checkpoint cadence: the `max_runtime` overshoot is bounded
/// by one chunk, the same bound the capped path already accepts. Chunk shape
/// never affects results (per-history Welford; seeds keyed to the global
/// particle index).
pub(crate) fn uncapped_chunk_size(batch: usize) -> usize {
    match batch {
        0 => 100,
        1 => 1_000,
        2 => 10_000,
        _ => 100_000,
    }
}

// ============================================================================
// TESTS
// ============================================================================

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::cell::Cell;

    #[test]
    fn format_secs_picks_a_sensible_unit() {
        // Sub-second → ms (rounded), no fractional ms.
        assert_eq!(format_secs(0.001), "1 ms");
        assert_eq!(format_secs(0.0234), "23 ms");
        assert_eq!(format_secs(0.9999), "1000 ms");
        // < 60s → "X.Xs".
        assert_eq!(format_secs(1.0), "1.0s");
        assert_eq!(format_secs(12.345), "12.3s");
        assert_eq!(format_secs(59.9), "59.9s");
        // < 1h → "Mm SSs".
        assert_eq!(format_secs(60.0), "1m 00s");
        assert_eq!(format_secs(125.0), "2m 05s");
        // ≥ 1h → "Hh MMm".
        assert_eq!(format_secs(3600.0), "1h 00m");
        assert_eq!(format_secs(7325.0), "2h 02m");
        // Edge: negative / NaN → "?".
        assert_eq!(format_secs(-1.0), "?");
        assert_eq!(format_secs(f64::NAN), "?");
    }

    #[test]
    fn runtime_budget_exhausted_predicate() {
        // No budget set: never exhausted, regardless of elapsed time.
        assert!(!runtime_budget_exhausted(None, 0.0));
        assert!(!runtime_budget_exhausted(None, 1e9));
        // Budget set: exhausted exactly once elapsed reaches it.
        assert!(!runtime_budget_exhausted(Some(5.0), 4.999));
        assert!(runtime_budget_exhausted(Some(5.0), 5.0)); // boundary is inclusive
        assert!(runtime_budget_exhausted(Some(5.0), 5.001));
        // Degenerate zero budget: exhausted at the first checkpoint.
        assert!(runtime_budget_exhausted(Some(0.0), 0.0));
    }

    use crate::geo::{BoundaryType, Surface};
    use crate::geo::{HalfspaceType, Region};
    use crate::transport::handle_photon_collision;
    use yamc_materials::material::Material;
    use yamc_particle::particle::ParticleType;
    use yamc_source::distribution::angular::AngularDistribution;
    use yamc_source::distribution::energy::Discrete;
    use yamc_source::distribution::spatial::Point;
    use yamc_source::source::{
        ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
    };

    /// `use_decay_photons` puts photons in flight, so the predicate that gates
    /// photon data has to count it. It did not.
    ///
    /// The gap was invisible from Python, whose constructor refuses
    /// `use_decay_photons` without `transport_secondary_photons`, so the two
    /// flags can never disagree there. From Rust, which has no such
    /// validation, it was a trap: the model reported no photons, so
    /// `ensure_photon_data_for_gpu` returned before its missing-data check,
    /// and the coupled path then panicked where that check exists to produce a
    /// clean error. Issue #43.
    ///
    /// No nuclear data is read, so this runs anywhere.
    #[test]
    fn a_decay_photon_model_reports_photons() {
        let sphere = Arc::new(Surface::sphere(
            0.0,
            0.0,
            0.0,
            10.0,
            Some(1),
            Some(BoundaryType::Vacuum),
        ));
        let mat = Material::new(
            HashMap::from([("Fe56".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(7.874),
        )
        .unwrap();
        let region = Region::new_from_halfspace(HalfspaceType::Below(sphere));
        let cell = Cell::new(Some(1), region, None, Some(0));
        let geometry = Geometry::new(vec![cell], vec![Arc::new(mat)]).unwrap();
        let mut model = Model::new(geometry, vec![], vec![]);

        // Neither flag, no photon source: nothing in flight, nothing to load.
        assert!(!model.has_photons());
        assert!(model.required_elements().is_empty());

        // Decay photons alone. This is the state Python forbids and Rust
        // allows, and the one the predicate used to miss.
        model.use_decay_photons = true;
        assert!(model.has_photons());
        assert_eq!(model.required_elements(), vec!["Fe".to_string()]);
    }

    /// Smoke test: run photon transport through an Fe sphere.
    /// Verifies the transport loop doesn't crash and particles are processed.
    #[test]
    fn test_transport_secondary_photons_fe_sphere() {
        // Skip if test data not available
        if !std::path::Path::new("tests/Fe.arrow").exists()
            || !std::path::Path::new("tests/Fe56.arrow").exists()
        {
            eprintln!("Skipping test_transport_secondary_photons_fe_sphere: test data not found");
            return;
        }

        // Configure global Config so ensure_nuclides_loaded() can find Fe56
        {
            let mut config = yamc_nuclide::config::Config::global();
            config.set_cross_section("Fe56", Some("tests/Fe56.arrow"));
        }

        // Create a sphere of Fe with vacuum boundary
        let sphere = Arc::new(Surface::sphere(
            0.0,
            0.0,
            0.0,
            10.0,
            Some(1),
            Some(BoundaryType::Vacuum),
        ));

        // Fe material
        let mut mat = Material::new(
            HashMap::from([("Fe56".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(7.874),
        )
        .unwrap();
        mat.set_temperature("294");
        mat.photon_data_paths
            .insert("Fe".to_string(), "tests/Fe.arrow".to_string());

        // Cell: inside sphere (Below = negative halfspace = inside)
        let region = Region::new_from_halfspace(HalfspaceType::Below(sphere.clone()));
        let cell = Cell::new(Some(1), region, None, Some(0));

        let geometry = Geometry::new(vec![cell], vec![Arc::new(mat)]).unwrap();

        // Photon source at origin, 1 MeV
        let source = ParticleSource::Photon(Source {
            space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
            angle: AngularDistribution::Isotropic,
            energy: SourceEnergyDistribution::Discrete(
                Discrete::new(vec![1_000_000.0], vec![1.0]).unwrap(),
            ),
            strength: 1.0,
        });

        // transport_secondary_photons is auto-enabled since source particle type is Photon
        let mut model = Model::new(geometry, vec![source], vec![]);
        model.photon_cutoff_energy = 1000.0;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(100),
                ..Default::default()
            })
            .unwrap();

        // If we get here without panic, photon transport works
        assert!(
            model.last_elapsed_secs.is_some(),
            "Model should have run timing data"
        );
        assert!(
            model.last_particles_per_second.unwrap_or(0) > 0,
            "Should have processed particles"
        );
    }

    /// The particle-free auto-N ray-marcher integrates `Sigma_t * length`: a ray
    /// from the centre of a homogeneous Fe sphere straight out must give
    /// `tau = Sigma_t(E) * radius` (then vacuum contributes nothing).
    #[test]
    fn integrate_optical_depth_matches_sigma_t_times_length() {
        if !std::path::Path::new("tests/Fe56.arrow").exists() {
            eprintln!("Skipping: tests/Fe56.arrow not found");
            return;
        }
        let r = 10.0;
        let e = 1.0e6;
        let sphere = Arc::new(Surface::sphere(
            0.0,
            0.0,
            0.0,
            r,
            Some(1),
            Some(BoundaryType::Vacuum),
        ));
        let mut mat = Material::new(
            HashMap::from([("Fe56".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(7.874),
        )
        .unwrap();
        mat.set_temperature("294");
        let mut nuclide_map = HashMap::new();
        nuclide_map.insert("Fe56".to_string(), "tests/Fe56.arrow".to_string());
        mat.read_nuclear_data(&nuclide_map, None).unwrap();
        mat.calculate_macroscopic_xs(&vec![1], true);
        let sigma_t = mat.lookup_xs_by_mt(1, e);
        assert!(sigma_t > 0.0, "Sigma_t should be positive");

        let region = Region::new_from_halfspace(HalfspaceType::Below(sphere.clone()));
        let cell = Cell::new(Some(1), region, None, Some(0));
        let geometry = Geometry::new(vec![cell], vec![Arc::new(mat)]).unwrap();
        let model = Model::new(geometry, vec![], vec![]);

        // Centre -> +x: `r` cm of Fe, then vacuum. tau == Sigma_t * r.
        let tau = model.integrate_optical_depth([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], r, e, false);
        let expected = sigma_t * r;
        assert!(
            (tau - expected).abs() < 1e-6 * expected.max(1.0),
            "ray-trace tau {tau} != Sigma_t*r {expected}"
        );

        // A ray shorter than the radius integrates only that length.
        let half =
            model.integrate_optical_depth([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], r / 2.0, e, false);
        assert!((half - sigma_t * r / 2.0).abs() < 1e-6 * expected.max(1.0));
    }

    /// Test handle_photon_collision directly with Fe photon data.
    #[test]
    fn test_handle_photon_collision_fe() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        if !std::path::Path::new("tests/Fe.arrow").exists() {
            eprintln!("Skipping: tests/Fe.arrow not found");
            return;
        }

        // Set up Fe material with photon data
        let mut mat = Material::new(
            HashMap::from([("Fe56".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(7.874),
        )
        .unwrap();
        mat.set_temperature("294");

        // Load nuclide data directly via read_nuclear_data (avoids global Config dependency)
        let mut nuclide_map = HashMap::new();
        nuclide_map.insert("Fe56".to_string(), "tests/Fe56.arrow".to_string());
        mat.read_nuclear_data(&nuclide_map, None).unwrap();
        mat.calculate_macroscopic_xs(&vec![1], true);

        let mut photon_paths = HashMap::new();
        photon_paths.insert("Fe".to_string(), "tests/Fe.arrow".to_string());
        mat.init_photon_data(&photon_paths).unwrap();

        let mut rng = StdRng::seed_from_u64(42);
        let n = 1000;
        let mut coherent = 0;
        let mut incoherent = 0;
        let mut photoelectric = 0;
        let mut pair_prod = 0;

        for _ in 0..n {
            let mut particle = yamc_particle::particle::Particle::new(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                1_000_000.0, // 1 MeV
            );
            particle.particle_type = ParticleType::Photon;
            particle.weight = 1.0;
            particle.alive = true;

            let macro_xs = mat.calculate_photon_xs(particle.energy);
            let mut bank = ParticleBank::new();

            let (mt, _bank_second) = handle_photon_collision(
                &mut particle,
                &mat,
                &macro_xs,
                &mut bank,
                1000.0,
                &mut rng,
            );

            match mt {
                502 => {
                    coherent += 1;
                    assert!(particle.alive, "Coherent: particle should stay alive");
                    assert!(
                        (particle.energy - 1_000_000.0).abs() < 1e-6,
                        "Coherent: no energy change"
                    );
                }
                504 => {
                    incoherent += 1;
                    assert!(particle.alive, "Incoherent: particle should stay alive");
                    assert!(
                        particle.energy < 1_000_000.0,
                        "Incoherent: energy should decrease"
                    );
                    assert!(
                        particle.energy > 0.0,
                        "Incoherent: energy should be positive"
                    );
                }
                mt if mt >= 534 => {
                    photoelectric += 1;
                    assert!(
                        !particle.alive,
                        "Photoelectric: particle should be absorbed"
                    );
                }
                515 => {
                    pair_prod += 1;
                    assert!(
                        !particle.alive,
                        "Pair production: particle should be absorbed"
                    );
                    // Should have banked 2 annihilation photons
                    assert_eq!(
                        bank.len(),
                        2,
                        "Pair production should bank 2 annihilation photons"
                    );
                }
                _ => {}
            }
        }

        // At 1 MeV for Fe, incoherent (Compton) should dominate
        assert!(
            incoherent > coherent,
            "At 1 MeV, Compton should dominate over Rayleigh: {incoherent} vs {coherent}"
        );
        // All reactions should sum to n
        let total = coherent + incoherent + photoelectric + pair_prod;
        assert_eq!(total, n, "All collisions should produce a valid reaction");
    }

    /// Test coupled n-gamma: neutron source in Fe sphere with transport_secondary_photons=true.
    /// Verifies that secondary photons are produced from neutron collisions and transported.
    #[test]
    fn test_coupled_neutron_gamma_fe_sphere() {
        if !std::path::Path::new("tests/Fe.arrow").exists()
            || !std::path::Path::new("tests/Fe56.arrow").exists()
        {
            eprintln!("Skipping test_coupled_neutron_gamma_fe_sphere: test data not found");
            return;
        }

        // Configure global Config for Fe56 nuclide data
        {
            let mut config = yamc_nuclide::config::Config::global();
            config.set_cross_section("Fe56", Some("tests/Fe56.arrow"));
        }

        // Create a sphere of Fe
        let sphere = Arc::new(Surface::sphere(
            0.0,
            0.0,
            0.0,
            10.0,
            Some(1),
            Some(BoundaryType::Vacuum),
        ));

        let mut mat = Material::new(
            HashMap::from([("Fe56".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(7.874),
        )
        .unwrap();
        mat.set_temperature("294");
        mat.photon_data_paths
            .insert("Fe".to_string(), "tests/Fe.arrow".to_string());

        let region = Region::new_from_halfspace(HalfspaceType::Below(sphere.clone()));
        let cell = Cell::new(Some(1), region, None, Some(0));
        let geometry = Geometry::new(vec![cell], vec![Arc::new(mat)]).unwrap();

        // NEUTRON source at origin, 14 MeV (high energy to ensure photon production)
        let source = ParticleSource::Neutron(Source {
            space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
            angle: AngularDistribution::Isotropic,
            energy: SourceEnergyDistribution::Discrete(
                Discrete::new(vec![14_000_000.0], vec![1.0]).unwrap(),
            ),
            strength: 1.0,
        });

        let mut model = Model::new(geometry, vec![source], vec![]);
        model.transport_secondary_photons = true;
        model.photon_cutoff_energy = 1000.0;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(100),
                ..Default::default()
            })
            .unwrap();

        // If we get here without panic, coupled n-gamma transport works
        assert!(
            model.last_elapsed_secs.is_some(),
            "Model should have run timing data"
        );
        assert!(
            model.last_particles_per_second.unwrap_or(0) > 0,
            "Should have processed particles"
        );
    }

    /// Test that lost particles are collected (not panicked) when geometry has a gap.
    ///
    /// Creates a void sphere (no material) with a transmission boundary and no
    /// surrounding cell. Particles cross the sphere surface and get lost.
    #[test]
    fn test_lost_particles_collected() {
        use crate::geo::SurfaceKind;

        // Inner sphere: transmission boundary (particles cross into the gap)
        let sphere = Arc::new(Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 5.0,
            },
            boundary: BoundaryType::Transmission,
            name: None,
        });

        // Single cell: inside the sphere, void (no material)
        let region = Region::new_from_halfspace(HalfspaceType::Below(sphere.clone()));
        let cell = Cell::new(Some(1), region, Some("inner_void".to_string()), None);
        let geometry = Geometry::new(vec![cell], Vec::new()).unwrap();

        // Monodirectional source: shoots particles in +z direction
        let source = ParticleSource::Neutron(Source {
            space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
            angle: AngularDistribution::new_monodirectional(0.0, 0.0, 1.0),
            energy: SourceEnergyDistribution::Discrete(
                Discrete::new(vec![1_000_000.0], vec![1.0]).unwrap(),
            ),
            strength: 1.0,
        });

        let mut model = Model::new(geometry, vec![source], vec![]);
        model.max_lost_particles = 10; // Allow up to 10
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(5),
                threads: Some(1), // single thread for determinism
                ..Default::default()
            })
            .unwrap();

        // All 5 particles should be lost (they cross the sphere into nothing)
        assert_eq!(
            model.lost_particles.len(),
            5,
            "Expected 5 lost particles, got {}",
            model.lost_particles.len()
        );

        // Check diagnostic info is populated
        for lp in &model.lost_particles {
            assert_eq!(lp.particle_type, ParticleType::Neutron);
            assert_eq!(lp.last_cell_id, Some(1));
            assert_eq!(lp.last_cell_name.as_deref(), Some("inner_void"));
            assert_eq!(lp.surface_id, Some(1));
            assert!(lp.energy > 0.0);
            // Position should be near the sphere surface (r ≈ 5.0)
            let r =
                (lp.position[0].powi(2) + lp.position[1].powi(2) + lp.position[2].powi(2)).sqrt();
            assert!(
                (r - 5.0).abs() < 0.01,
                "Lost particle should be near sphere surface, got r={}",
                r
            );
        }
    }

    /// Test that exceeding max_lost_particles causes a panic.
    #[test]
    #[should_panic(expected = "Maximum lost particles exceeded")]
    fn test_max_lost_particles_exceeded() {
        use crate::geo::SurfaceKind;

        let sphere = Arc::new(Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 5.0,
            },
            boundary: BoundaryType::Transmission,
            name: None,
        });

        let region = Region::new_from_halfspace(HalfspaceType::Below(sphere.clone()));
        let cell = Cell::new(Some(1), region, None, None);
        let geometry = Geometry::new(vec![cell], Vec::new()).unwrap();

        let source = ParticleSource::Neutron(Source {
            space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
            angle: AngularDistribution::new_monodirectional(0.0, 0.0, 1.0),
            energy: SourceEnergyDistribution::Discrete(
                Discrete::new(vec![1_000_000.0], vec![1.0]).unwrap(),
            ),
            strength: 1.0,
        });

        // Allow only 3 lost particles, but run 10
        let mut model = Model::new(geometry, vec![source], vec![]);
        model.max_lost_particles = 3;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(10),
                threads: Some(1),
                ..Default::default()
            })
            .unwrap(); // Should panic
    }

    #[test]
    fn chunk_schedule_sums_to_total_caps_size_and_is_even() {
        use super::{derive_chunk_count, derive_chunk_size, MAX_PARTICLES_PER_CPU_CHUNK};
        // Edge totals.
        assert_eq!(derive_chunk_count(0), 0);
        assert_eq!(derive_chunk_count(1), 1);
        assert_eq!(derive_chunk_size(1, 1, 0), 1);

        for &total in &[7usize, 10, 100, 999, 1000, 12_345, 1_000_000, 100_000_007] {
            let n = derive_chunk_count(total);
            assert!(n >= 1, "total={total}");
            let sizes: Vec<usize> = (0..n).map(|b| derive_chunk_size(total, n, b)).collect();
            assert!(
                sizes.iter().all(|&s| s > 0),
                "non-positive chunk, total={total}"
            );
            // Sums to exactly the total (no truncation, no overshoot).
            assert_eq!(
                sizes.iter().sum::<usize>(),
                total,
                "schedule must sum to total ({total})"
            );
            // No chunk exceeds the cap, and chunks are evenly sized.
            assert!(
                sizes.iter().all(|&s| s <= MAX_PARTICLES_PER_CPU_CHUNK),
                "chunk exceeds cap for {total}"
            );
            let (lo, hi) = (*sizes.iter().min().unwrap(), *sizes.iter().max().unwrap());
            assert!(hi - lo <= 1, "chunks not even for {total}: {lo}..={hi}");
        }
        // Short runs still use exactly TARGET_CHUNKS (=10) for the snapshot cadence.
        assert_eq!(derive_chunk_count(100), 10);
        // Large runs cap chunk size instead of ballooning it: the count scales
        // with the total (total / cap chunks), not a fixed 10.
        assert_eq!(
            derive_chunk_count(1_000_000),
            1_000_000 / MAX_PARTICLES_PER_CPU_CHUNK
        );
        // A very large total is handled without a giant chunk or a giant Vec
        // (issue #193): the count scales up and every chunk stays within the
        // cap. The loop evaluates sizes lazily, so nothing this large is built.
        // 1e12 only fits a 64-bit usize; a 32-bit target (wasm32) cannot even
        // represent a run this large, so gate the stress value to 64-bit.
        #[cfg(target_pointer_width = "64")]
        {
            let huge = 1_000_000_000_000usize; // 1e12
            let n = derive_chunk_count(huge);
            assert_eq!(n, huge.div_ceil(MAX_PARTICLES_PER_CPU_CHUNK));
            assert!(derive_chunk_size(huge, n, 0) <= MAX_PARTICLES_PER_CPU_CHUNK);
            assert!(derive_chunk_size(huge, n, n - 1) <= MAX_PARTICLES_PER_CPU_CHUNK);
        }
    }

    /// A many-source model is what a parametric plasma source builds, so the
    /// fast path must pick exactly the source the walk would have.
    #[test]
    fn selector_sampling_matches_the_linear_walk() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use yamc_source::source::SourceSelector;

        // Uneven strengths, including a zero-strength source that must never
        // be selected, and a repeated value so ties resolve the same way.
        let strengths = [3.0, 0.0, 1.0, 1.0, 7.5, 0.25];
        let sources: Vec<ParticleSource> = strengths
            .iter()
            .enumerate()
            .map(|(index, strength)| {
                ParticleSource::Neutron(Source {
                    space: SourceSpatialDistribution::Point(Point::new([index as f64, 0.0, 0.0])),
                    angle: AngularDistribution::Isotropic,
                    energy: SourceEnergyDistribution::Discrete(
                        Discrete::new(vec![1.0e6], vec![1.0]).unwrap(),
                    ),
                    strength: *strength,
                })
            })
            .collect();
        let model = Model::new(
            Geometry::new(
                vec![Cell::new(
                    Some(1),
                    Region::new_from_halfspace(HalfspaceType::Below(Arc::new(Surface::sphere(
                        0.0,
                        0.0,
                        0.0,
                        10.0,
                        Some(1),
                        Some(BoundaryType::Vacuum),
                    )))),
                    None,
                    None,
                )],
                vec![],
            )
            .unwrap(),
            sources,
            vec![],
        );

        let selector = SourceSelector::new(&model.sources);
        assert_eq!(selector.total(), 12.75);
        let mut walk_rng = StdRng::seed_from_u64(5);
        let mut selector_rng = StdRng::seed_from_u64(5);
        for _ in 0..2000 {
            let walked = model.sample_source(&mut walk_rng);
            let selected = model.sample_source_with(&selector, &mut selector_rng);
            assert_eq!(walked.position, selected.position);
            // The zero-strength source sits at x = 1 and must never emit.
            assert_ne!(selected.position[0], 1.0);
        }
    }

    #[test]
    fn uncapped_chunk_size_ramps_then_steadies() {
        // Front-loaded so the first stop-condition checkpoints (max_runtime /
        // convergence) arrive after only 100 then 1_100 cumulative histories,
        // then a steady 100_000 chunk for throughput.
        assert_eq!(super::uncapped_chunk_size(0), 100);
        assert_eq!(super::uncapped_chunk_size(1), 1_000);
        assert_eq!(super::uncapped_chunk_size(2), 10_000);
        for batch in 3..20 {
            assert_eq!(super::uncapped_chunk_size(batch), 100_000);
        }
        // Every chunk is positive, so an uncapped run always makes progress.
        assert!((0..50).all(|b| super::uncapped_chunk_size(b) > 0));
    }
}
