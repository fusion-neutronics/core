//! Particle tracking system for YAMC.
//!
//! The [`Tracker`] trait is an **event collector**, not a transport-algorithm
//! dispatcher. It records discrete events (birth, collision, surface crossing,
//! termination) that occurred during particle transport so they can be replayed
//! for debugging, visualisation, or off-line analysis. Implementations do not
//! decide *how* a particle moves or interacts -- that lives in
//! `transport_particle`.
//!
//! Two implementations are provided and selected via monomorphisation, so the
//! cost is zero when tracking is disabled:
//! - [`NoOpTracker`] -- every method is empty; the optimiser inlines the calls
//!   to nothing. Used in production runs where no track output is requested.
//! - [`RealTracker`] -- records events into a [`TrackStorage`] buffer; used when
//!   the user has asked for per-particle history output.
//!
//! Adding a new transport mode (e.g. Woodcock tracking) does not require a
//! change to this trait -- the new transport code calls the same
//! `record_collision` / `record_surface_crossing` methods.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};

/// Which particle histories to capture full event tracks for.
///
/// Histories are identified by their **global index**: a 0-based counter over
/// every source history in the whole run, independent of how particles are
/// grouped into chunks/threads. Membership is therefore deterministic -- the
/// captured set does not depend on thread scheduling.
///
/// The Python `capture_tracks=` argument maps onto this:
/// - `int N`        -> [`HistorySelection::first`] (global indices `0..N`)
/// - `range(a,b,s)` -> [`HistorySelection::Range`] (half-open, strided)
/// - `[i, j, ...]`  -> [`HistorySelection::Set`] (exactly those indices)
/// - `'all'`        -> [`HistorySelection::All`]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistorySelection {
    /// Capture every history (potentially unbounded memory).
    All,
    /// Capture the half-open index range `[start, stop)` with the given
    /// positive stride (`step >= 1`).
    Range { start: u64, stop: u64, step: u64 },
    /// Capture exactly these global history indices.
    Set(BTreeSet<u64>),
}

impl HistorySelection {
    /// Capture the first `n` histories (global indices `0..n`).
    pub fn first(n: u64) -> Self {
        HistorySelection::Range {
            start: 0,
            stop: n,
            step: 1,
        }
    }

    /// Is the history with this global `index` captured?
    pub fn contains(&self, index: u64) -> bool {
        match self {
            HistorySelection::All => true,
            HistorySelection::Range { start, stop, step } => {
                index >= *start && index < *stop && (index - *start).is_multiple_of(*step)
            }
            HistorySelection::Set(set) => set.contains(&index),
        }
    }
}

/// Event types during transport
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackEventType {
    /// Particle created from the source distribution
    SourceBirth,
    /// Particle created from a reaction (fission, n2n, etc.)
    SecondaryBirth,
    /// Collision occurred (scattering, absorption, fission)
    Collision,
    /// Particle crossed a surface
    SurfaceCrossing,
    /// Particle was absorbed
    Absorption,
    /// Particle killed by weight-cutoff Russian roulette (survival biasing)
    RussianRoulette,
    /// Particle leaked from geometry (vacuum boundary)
    Leak,
}

impl std::fmt::Display for TrackEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackEventType::SourceBirth => write!(f, "source_birth"),
            TrackEventType::SecondaryBirth => write!(f, "secondary_birth"),
            TrackEventType::Collision => write!(f, "collision"),
            TrackEventType::SurfaceCrossing => write!(f, "surface_crossing"),
            TrackEventType::Absorption => write!(f, "absorption"),
            TrackEventType::RussianRoulette => write!(f, "russian_roulette"),
            TrackEventType::Leak => write!(f, "leak"),
        }
    }
}

/// A single event in a particle's track
#[derive(Debug, Clone)]
pub struct TrackEvent {
    /// Unique ID across all particles
    pub particle_id: u64,
    /// Parent particle ID (None for source particles)
    pub parent_id: Option<u64>,
    /// Generation (0 = source, 1+ = secondary)
    pub generation: u32,
    /// Batch number
    pub batch: usize,
    /// History within batch
    pub history: usize,
    /// Type of event
    pub event_type: TrackEventType,
    /// Position where event occurred [x, y, z] in cm
    pub position: [f64; 3],
    /// Direction at event [u, v, w] unit vector
    pub direction: [f64; 3],
    /// Energy before event (eV)
    pub energy_in: f64,
    /// Energy after event (eV)
    pub energy_out: f64,
    /// Statistical weight
    pub weight: f64,
    /// Cell ID where event occurred
    pub cell_id: Option<u32>,
    /// MT number for collisions
    pub reaction_mt: Option<i32>,
    /// Target nuclide for collisions
    pub nuclide: Option<String>,
    /// Birth reaction type (e.g., "source", "fission", "n,2n")
    pub birth_reaction: Option<String>,
    /// Distribution type used for angle-energy sampling
    pub distribution: Option<String>,
    /// Energy distribution type (when using UncorrelatedAngleEnergy)
    pub energy_dist: Option<String>,
}

impl TrackEvent {
    /// Create a new event with minimal required fields
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        particle_id: u64,
        parent_id: Option<u64>,
        generation: u32,
        batch: usize,
        history: usize,
        event_type: TrackEventType,
        position: [f64; 3],
        direction: [f64; 3],
        energy_in: f64,
        energy_out: f64,
        weight: f64,
    ) -> Self {
        TrackEvent {
            particle_id,
            parent_id,
            generation,
            batch,
            history,
            event_type,
            position,
            direction,
            energy_in,
            energy_out,
            weight,
            cell_id: None,
            reaction_mt: None,
            nuclide: None,
            birth_reaction: None,
            distribution: None,
            energy_dist: None,
        }
    }
}

/// A complete track for a single particle
#[derive(Debug, Clone, Default)]
pub struct ParticleTrack {
    /// All events in this particle's history
    pub events: Vec<TrackEvent>,
    /// Particle ID
    pub particle_id: u64,
    /// Parent particle ID (None for source particles)
    pub parent_id: Option<u64>,
    /// Generation (0 = source, 1+ = secondary)
    pub generation: u32,
}

/// Storage for all collected tracks
#[derive(Debug, Clone, Default)]
pub struct TrackStorage {
    /// All particle tracks collected during simulation
    pub tracks: Vec<ParticleTrack>,
}

impl TrackStorage {
    pub fn new() -> Self {
        TrackStorage { tracks: Vec::new() }
    }

    /// Total number of events across all tracks
    pub fn total_events(&self) -> usize {
        self.tracks.iter().map(|t| t.events.len()).sum()
    }
}

/// Metadata tracked per particle (only allocated when tracking enabled)
#[derive(Debug, Clone)]
pub struct ParticleTrackingInfo {
    pub track_id: u64,
    pub parent_id: Option<u64>,
    pub generation: u32,
    pub birth_reaction: String,
}

impl Default for ParticleTrackingInfo {
    fn default() -> Self {
        ParticleTrackingInfo {
            track_id: 0,
            parent_id: None,
            generation: 0,
            birth_reaction: "source".to_string(),
        }
    }
}

/// Collects events that occur during particle transport.
///
/// This is a passive event sink -- implementations record what happened, they
/// do not drive transport. The transport loop (`transport_particle`)
/// owns the algorithmic decisions (distance sampling, surface crossings,
/// reaction sampling) and calls the `record_*` methods to publish the events
/// it produced. The same trait is used regardless of transport algorithm
/// (e.g. surface tracking, Woodcock tracking).
///
/// Monomorphisation gives zero overhead when tracking is disabled -- see
/// [`NoOpTracker`].
#[allow(clippy::too_many_arguments)]
pub trait Tracker: Send {
    /// Create a new tracker that captures the given selection of histories.
    fn new_with_selection(selection: HistorySelection) -> Self;

    /// Signal the start of a history with the given global `history` index.
    fn start_history(&mut self, batch: usize, history: usize);

    /// Signal end of current history
    fn finish_history(&mut self);

    /// Record a birth event (source or secondary)
    fn record_birth(
        &mut self,
        particle_id: u64,
        parent_id: Option<u64>,
        generation: u32,
        position: [f64; 3],
        direction: [f64; 3],
        energy: f64,
        weight: f64,
        cell_id: Option<u32>,
        birth_reaction: &str,
    );

    /// Record a collision event
    fn record_collision(
        &mut self,
        particle_id: u64,
        parent_id: Option<u64>,
        generation: u32,
        position: [f64; 3],
        direction: [f64; 3],
        energy_in: f64,
        energy_out: f64,
        weight: f64,
        cell_id: Option<u32>,
        mt: i32,
        nuclide: &str,
        distribution: Option<&str>,
        energy_dist: Option<&str>,
    );

    /// Record a surface crossing event
    fn record_surface_crossing(
        &mut self,
        particle_id: u64,
        parent_id: Option<u64>,
        generation: u32,
        position: [f64; 3],
        direction: [f64; 3],
        energy: f64,
        weight: f64,
        cell_id: Option<u32>,
    );

    /// Record a termination event (absorption or leak)
    fn record_termination(
        &mut self,
        particle_id: u64,
        parent_id: Option<u64>,
        generation: u32,
        position: [f64; 3],
        direction: [f64; 3],
        energy: f64,
        weight: f64,
        cell_id: Option<u32>,
        event_type: TrackEventType,
    );

    /// Get the next unique particle ID
    fn next_particle_id(&mut self) -> u64;

    /// Check if tracking is enabled (for conditional code paths)
    fn should_track(&self) -> bool;

    /// Get collected results (consumes the tracker)
    fn into_storage(self) -> TrackStorage;
}

/// No-op tracker - all methods compile to nothing via inlining
#[derive(Debug, Clone, Default)]
pub struct NoOpTracker;

impl NoOpTracker {
    pub fn new() -> Self {
        NoOpTracker
    }
}

impl Tracker for NoOpTracker {
    #[inline(always)]
    fn new_with_selection(_selection: HistorySelection) -> Self {
        NoOpTracker
    }

    #[inline(always)]
    fn start_history(&mut self, _batch: usize, _history: usize) {}

    #[inline(always)]
    fn finish_history(&mut self) {}

    #[inline(always)]
    fn record_birth(
        &mut self,
        _particle_id: u64,
        _parent_id: Option<u64>,
        _generation: u32,
        _position: [f64; 3],
        _direction: [f64; 3],
        _energy: f64,
        _weight: f64,
        _cell_id: Option<u32>,
        _birth_reaction: &str,
    ) {
    }

    #[inline(always)]
    fn record_collision(
        &mut self,
        _particle_id: u64,
        _parent_id: Option<u64>,
        _generation: u32,
        _position: [f64; 3],
        _direction: [f64; 3],
        _energy_in: f64,
        _energy_out: f64,
        _weight: f64,
        _cell_id: Option<u32>,
        _mt: i32,
        _nuclide: &str,
        _distribution: Option<&str>,
        _energy_dist: Option<&str>,
    ) {
    }

    #[inline(always)]
    fn record_surface_crossing(
        &mut self,
        _particle_id: u64,
        _parent_id: Option<u64>,
        _generation: u32,
        _position: [f64; 3],
        _direction: [f64; 3],
        _energy: f64,
        _weight: f64,
        _cell_id: Option<u32>,
    ) {
    }

    #[inline(always)]
    fn record_termination(
        &mut self,
        _particle_id: u64,
        _parent_id: Option<u64>,
        _generation: u32,
        _position: [f64; 3],
        _direction: [f64; 3],
        _energy: f64,
        _weight: f64,
        _cell_id: Option<u32>,
        _event_type: TrackEventType,
    ) {
    }

    #[inline(always)]
    fn next_particle_id(&mut self) -> u64 {
        0
    }

    #[inline(always)]
    fn should_track(&self) -> bool {
        false
    }

    #[inline(always)]
    fn into_storage(self) -> TrackStorage {
        TrackStorage::new()
    }
}

/// Real tracker that collects events
#[derive(Debug, Clone)]
pub struct RealTracker {
    /// All collected tracks
    tracks: Vec<ParticleTrack>,
    /// Current track being built
    current_track: Option<ParticleTrack>,
    /// Current batch number
    current_batch: usize,
    /// Current history number
    current_history: usize,
    /// Next particle ID (thread-local counter)
    next_id: u64,
    /// Which global history indices to capture
    selection: HistorySelection,
    /// Whether the current history is being captured (set per history from
    /// `selection`)
    active: bool,
}

impl RealTracker {
    /// Create a new real tracker capturing the given selection of histories
    pub fn new(selection: HistorySelection) -> Self {
        RealTracker {
            tracks: Vec::new(),
            current_track: None,
            current_batch: 0,
            current_history: 0,
            next_id: 0,
            selection,
            active: false,
        }
    }

    /// Create from an existing global ID counter for thread-safety
    pub fn with_starting_id(selection: HistorySelection, starting_id: u64) -> Self {
        RealTracker {
            tracks: Vec::new(),
            current_track: None,
            current_batch: 0,
            current_history: 0,
            next_id: starting_id,
            selection,
            active: false,
        }
    }
}

impl Tracker for RealTracker {
    fn new_with_selection(selection: HistorySelection) -> Self {
        RealTracker::new(selection)
    }

    fn start_history(&mut self, batch: usize, history: usize) {
        // Capture this history only if its global index is in the selection.
        // The decision is per-history (not monotonic), so it stays correct no
        // matter which thread processes which index.
        self.active = self.selection.contains(history as u64);
        if !self.active {
            return;
        }

        self.current_batch = batch;
        self.current_history = history;
        self.current_track = Some(ParticleTrack::default());
    }

    fn finish_history(&mut self) {
        if !self.active {
            return;
        }

        if let Some(track) = self.current_track.take() {
            if !track.events.is_empty() {
                self.tracks.push(track);
            }
        }
    }

    fn record_birth(
        &mut self,
        particle_id: u64,
        parent_id: Option<u64>,
        generation: u32,
        position: [f64; 3],
        direction: [f64; 3],
        energy: f64,
        weight: f64,
        cell_id: Option<u32>,
        birth_reaction: &str,
    ) {
        if !self.active {
            return;
        }

        let event_type = if generation == 0 {
            TrackEventType::SourceBirth
        } else {
            TrackEventType::SecondaryBirth
        };

        let mut event = TrackEvent::new(
            particle_id,
            parent_id,
            generation,
            self.current_batch,
            self.current_history,
            event_type,
            position,
            direction,
            energy,
            energy, // energy_out same as energy_in for birth
            weight,
        );
        event.cell_id = cell_id;
        event.birth_reaction = Some(birth_reaction.to_string());

        if let Some(track) = &mut self.current_track {
            // Update track metadata from first event
            if track.events.is_empty() {
                track.particle_id = particle_id;
                track.parent_id = parent_id;
                track.generation = generation;
            }
            track.events.push(event);
        }
    }

    fn record_collision(
        &mut self,
        particle_id: u64,
        parent_id: Option<u64>,
        generation: u32,
        position: [f64; 3],
        direction: [f64; 3],
        energy_in: f64,
        energy_out: f64,
        weight: f64,
        cell_id: Option<u32>,
        mt: i32,
        nuclide: &str,
        distribution: Option<&str>,
        energy_dist: Option<&str>,
    ) {
        if !self.active {
            return;
        }

        let mut event = TrackEvent::new(
            particle_id,
            parent_id,
            generation,
            self.current_batch,
            self.current_history,
            TrackEventType::Collision,
            position,
            direction,
            energy_in,
            energy_out,
            weight,
        );
        event.cell_id = cell_id;
        event.reaction_mt = Some(mt);
        event.nuclide = Some(nuclide.to_string());
        event.distribution = distribution.map(|s| s.to_string());
        event.energy_dist = energy_dist.map(|s| s.to_string());

        if let Some(track) = &mut self.current_track {
            track.events.push(event);
        }
    }

    fn record_surface_crossing(
        &mut self,
        particle_id: u64,
        parent_id: Option<u64>,
        generation: u32,
        position: [f64; 3],
        direction: [f64; 3],
        energy: f64,
        weight: f64,
        cell_id: Option<u32>,
    ) {
        if !self.active {
            return;
        }

        let mut event = TrackEvent::new(
            particle_id,
            parent_id,
            generation,
            self.current_batch,
            self.current_history,
            TrackEventType::SurfaceCrossing,
            position,
            direction,
            energy,
            energy,
            weight,
        );
        event.cell_id = cell_id;

        if let Some(track) = &mut self.current_track {
            track.events.push(event);
        }
    }

    fn record_termination(
        &mut self,
        particle_id: u64,
        parent_id: Option<u64>,
        generation: u32,
        position: [f64; 3],
        direction: [f64; 3],
        energy: f64,
        weight: f64,
        cell_id: Option<u32>,
        event_type: TrackEventType,
    ) {
        if !self.active {
            return;
        }

        let mut event = TrackEvent::new(
            particle_id,
            parent_id,
            generation,
            self.current_batch,
            self.current_history,
            event_type,
            position,
            direction,
            energy,
            energy,
            weight,
        );
        event.cell_id = cell_id;

        if let Some(track) = &mut self.current_track {
            track.events.push(event);
        }
    }

    fn next_particle_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn should_track(&self) -> bool {
        self.active
    }

    fn into_storage(mut self) -> TrackStorage {
        // Finalize any remaining track
        if let Some(track) = self.current_track.take() {
            if !track.events.is_empty() {
                self.tracks.push(track);
            }
        }
        TrackStorage {
            tracks: self.tracks,
        }
    }
}

/// Thread-safe wrapper for collecting tracks from parallel execution
#[derive(Debug)]
pub struct ThreadSafeTracker {
    /// Global particle ID counter (atomic for thread safety)
    global_id: AtomicU64,
    /// Which global history indices to capture
    selection: HistorySelection,
}

impl ThreadSafeTracker {
    pub fn new(selection: HistorySelection) -> Self {
        ThreadSafeTracker {
            global_id: AtomicU64::new(0),
            selection,
        }
    }

    /// Create a thread-local tracker with a unique ID range
    pub fn create_thread_tracker(&self) -> RealTracker {
        // Reserve a large block of IDs for this thread to avoid contention
        const ID_BLOCK_SIZE: u64 = 1_000_000;
        let starting_id = self.global_id.fetch_add(ID_BLOCK_SIZE, Ordering::Relaxed);
        RealTracker::with_starting_id(self.selection.clone(), starting_id)
    }

    /// Merge all thread-local trackers into a single storage
    pub fn merge_trackers(trackers: Vec<RealTracker>) -> TrackStorage {
        let mut all_tracks = Vec::new();
        for tracker in trackers {
            let storage = tracker.into_storage();
            all_tracks.extend(storage.tracks);
        }
        TrackStorage { tracks: all_tracks }
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_tracker() {
        let mut tracker = NoOpTracker::new();
        assert!(!tracker.should_track());
        assert_eq!(tracker.next_particle_id(), 0);

        tracker.start_history(0, 0);
        tracker.record_birth(
            0,
            None,
            0,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            1e6,
            1.0,
            Some(1),
            "source",
        );
        tracker.finish_history();

        let storage = tracker.into_storage();
        assert!(storage.tracks.is_empty());
    }

    #[test]
    fn test_real_tracker() {
        let mut tracker = RealTracker::new(HistorySelection::All);

        let pid = tracker.next_particle_id();
        assert_eq!(pid, 0);

        tracker.start_history(0, 0);
        assert!(tracker.should_track());
        tracker.record_birth(
            pid,
            None,
            0,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            1e6,
            1.0,
            Some(1),
            "source",
        );
        tracker.record_collision(
            pid,
            None,
            0,
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.707],
            1e6,
            9e5,
            1.0,
            Some(1),
            2,
            "Fe56",
            Some("UncorrelatedAngleEnergy"),
            Some("ContinuousTabular"),
        );
        tracker.finish_history();

        let storage = tracker.into_storage();
        assert_eq!(storage.tracks.len(), 1);
        assert_eq!(storage.tracks[0].events.len(), 2);

        let birth = &storage.tracks[0].events[0];
        assert_eq!(birth.event_type, TrackEventType::SourceBirth);
        assert_eq!(birth.birth_reaction, Some("source".to_string()));

        let collision = &storage.tracks[0].events[1];
        assert_eq!(collision.event_type, TrackEventType::Collision);
        assert_eq!(collision.reaction_mt, Some(2));
        assert_eq!(collision.nuclide, Some("Fe56".to_string()));
        assert_eq!(
            collision.distribution,
            Some("UncorrelatedAngleEnergy".to_string())
        );
        assert_eq!(collision.energy_dist, Some("ContinuousTabular".to_string()));
    }

    #[test]
    fn test_first_n_selection() {
        // first(2) captures global indices 0 and 1 only.
        let mut tracker = RealTracker::new(HistorySelection::first(2));

        for i in 0..5 {
            tracker.start_history(0, i);
            tracker.record_birth(
                i as u64,
                None,
                0,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                1e6,
                1.0,
                Some(1),
                "source",
            );
            tracker.finish_history();
        }

        let storage = tracker.into_storage();
        assert_eq!(storage.tracks.len(), 2); // Only 2 histories tracked
    }

    /// Run histories 0..n through a tracker and return the captured global
    /// history indices (in capture order).
    fn captured_histories(selection: HistorySelection, n: usize) -> Vec<usize> {
        let mut tracker = RealTracker::new(selection);
        for i in 0..n {
            tracker.start_history(0, i);
            tracker.record_birth(
                i as u64,
                None,
                0,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                1e6,
                1.0,
                Some(1),
                "source",
            );
            tracker.finish_history();
        }
        tracker
            .into_storage()
            .tracks
            .iter()
            .map(|t| t.events[0].history)
            .collect()
    }

    #[test]
    fn test_history_selection_contains() {
        assert!(HistorySelection::All.contains(123));

        let strided = HistorySelection::Range {
            start: 0,
            stop: 1000,
            step: 10,
        };
        assert!(strided.contains(0));
        assert!(strided.contains(990));
        assert!(!strided.contains(5));
        assert!(!strided.contains(1000)); // half-open: stop excluded

        let set = HistorySelection::Set([20u64, 30, 45].into_iter().collect());
        assert!(set.contains(30));
        assert!(!set.contains(31));
    }

    #[test]
    fn test_range_and_set_selection_capture_by_global_index() {
        // range(20, 30): histories 20..=29.
        assert_eq!(
            captured_histories(
                HistorySelection::Range {
                    start: 20,
                    stop: 30,
                    step: 1
                },
                50
            ),
            (20..30).collect::<Vec<_>>()
        );

        // range(0, 50, 10): every 10th history.
        assert_eq!(
            captured_histories(
                HistorySelection::Range {
                    start: 0,
                    stop: 50,
                    step: 10
                },
                50
            ),
            vec![0, 10, 20, 30, 40]
        );

        // Explicit index set, captured in ascending order.
        assert_eq!(
            captured_histories(
                HistorySelection::Set([20u64, 30, 45].into_iter().collect()),
                50
            ),
            vec![20, 30, 45]
        );
    }

    #[test]
    fn test_thread_safe_tracker() {
        let ts_tracker = ThreadSafeTracker::new(HistorySelection::All);

        let tracker1 = ts_tracker.create_thread_tracker();
        let tracker2 = ts_tracker.create_thread_tracker();

        // Each tracker should have different starting IDs
        assert_ne!(tracker1.next_id, tracker2.next_id);
    }
}
