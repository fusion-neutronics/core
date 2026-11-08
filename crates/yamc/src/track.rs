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

pub use yamc_particle::particle::ParticleType;

/// Event types during transport
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackEventType {
    /// Particle created from the source distribution
    SourceBirth,
    /// Particle popped from the bank as a descendant of another particle in
    /// the history (reaction secondary, secondary photon, weight-window split)
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

impl TrackEventType {
    /// Every variant, indexed by its discriminant (the polyline event code).
    pub const ALL: [TrackEventType; 7] = [
        TrackEventType::SourceBirth,
        TrackEventType::SecondaryBirth,
        TrackEventType::Collision,
        TrackEventType::SurfaceCrossing,
        TrackEventType::Absorption,
        TrackEventType::RussianRoulette,
        TrackEventType::Leak,
    ];
}

impl std::fmt::Display for TrackEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            TrackEventType::SourceBirth => "source_birth",
            TrackEventType::SecondaryBirth => "secondary_birth",
            TrackEventType::Collision => "collision",
            TrackEventType::SurfaceCrossing => "surface_crossing",
            TrackEventType::Absorption => "absorption",
            TrackEventType::RussianRoulette => "russian_roulette",
            TrackEventType::Leak => "leak",
        })
    }
}

/// A single event in a particle's track
#[derive(Debug, Clone)]
pub struct TrackEvent {
    /// Particle ID, unique within its history (0 = the source particle)
    pub particle_id: u64,
    /// ID of the particle that banked this one, in the same history (None
    /// for source particles)
    pub parent_id: Option<u64>,
    /// Generation (0 = source, 1+ = secondary)
    pub generation: u32,
    /// Particle species
    pub particle_type: ParticleType,
    /// Batch number
    pub batch: usize,
    /// Global history index
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
    /// Birth reaction ("source" or "secondary") on birth events
    pub birth_reaction: Option<String>,
    /// Distribution type used for angle-energy sampling
    pub distribution: Option<String>,
    /// Energy distribution type (when using UncorrelatedAngleEnergy)
    pub energy_dist: Option<String>,
}

/// The track of a single particle: its birth and every event of its walk.
#[derive(Debug, Clone, Default)]
pub struct ParticleTrack {
    /// All events in this particle's walk, starting with its birth
    pub events: Vec<TrackEvent>,
    /// Particle ID, unique within its history (0 = the source particle)
    pub particle_id: u64,
    /// ID of the particle that banked this one (None for source particles)
    pub parent_id: Option<u64>,
    /// Generation (0 = source, 1+ = secondary)
    pub generation: u32,
    /// Particle species
    pub particle_type: ParticleType,
    /// Global history index
    pub history: usize,
}

/// Storage for all collected tracks
#[derive(Debug, Clone, Default)]
pub struct TrackStorage {
    /// One track per particle, ordered by `(history, particle_id)` once merged
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

    /// Order tracks by `(history, particle_id)`, making the output independent
    /// of thread scheduling.
    pub fn sort(&mut self) {
        self.tracks.sort_by_key(|t| (t.history, t.particle_id));
    }

    /// Flatten the tracks into one polyline per track, for VTK export or
    /// plotting.
    pub fn polylines(&self) -> Polylines {
        let n_points = self.total_events();
        let n_lines = self.tracks.len();
        let mut out = Polylines {
            points: Vec::with_capacity(n_points),
            offsets: Vec::with_capacity(n_lines + 1),
            energy: Vec::with_capacity(n_points),
            weight: Vec::with_capacity(n_points),
            event_type: Vec::with_capacity(n_points),
            cell_id: Vec::with_capacity(n_points),
            particle_type: Vec::with_capacity(n_lines),
            particle_id: Vec::with_capacity(n_lines),
            parent_id: Vec::with_capacity(n_lines),
            generation: Vec::with_capacity(n_lines),
            history: Vec::with_capacity(n_lines),
        };
        out.offsets.push(0);
        for t in &self.tracks {
            for e in &t.events {
                out.points.push(e.position);
                out.energy.push(e.energy_out);
                out.weight.push(e.weight);
                out.event_type.push(e.event_type as u32);
                out.cell_id.push(e.cell_id.map_or(-1, i64::from));
            }
            out.offsets.push(out.points.len() as i64);
            out.particle_type.push(t.particle_type as u32);
            out.particle_id.push(t.particle_id);
            out.parent_id.push(t.parent_id.map_or(-1, |p| p as i64));
            out.generation.push(t.generation);
            out.history.push(t.history as u64);
        }
        out
    }
}

/// Tracks flattened into polylines: point `i` of line `j` is
/// `points[offsets[j] + i]`. Codes: `event_type` is the [`TrackEventType`]
/// discriminant, `particle_type` the [`ParticleType`] one (0 neutron, 1
/// photon); a missing `cell_id` / `parent_id` is -1.
#[derive(Debug, Clone, Default)]
pub struct Polylines {
    /// Event positions [x, y, z] in cm
    pub points: Vec<[f64; 3]>,
    /// Start of each line in `points`, plus the total point count (length
    /// = lines + 1)
    pub offsets: Vec<i64>,
    /// Per point: energy after the event (eV)
    pub energy: Vec<f64>,
    /// Per point: statistical weight
    pub weight: Vec<f64>,
    /// Per point: [`TrackEventType`] code
    pub event_type: Vec<u32>,
    /// Per point: cell ID, -1 if unknown
    pub cell_id: Vec<i64>,
    /// Per line: [`ParticleType`] code
    pub particle_type: Vec<u32>,
    /// Per line: particle ID within its history
    pub particle_id: Vec<u64>,
    /// Per line: parent particle ID, -1 for source particles
    pub parent_id: Vec<i64>,
    /// Per line: generation
    pub generation: Vec<u32>,
    /// Per line: global history index
    pub history: Vec<u64>,
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
/// Every particle walk opens with [`record_birth`](Tracker::record_birth),
/// which fixes the particle's identity; the other `record_*` events belong to
/// that particle until the next birth.
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

    /// Start a particle's track with its birth event (source or secondary).
    fn record_birth(
        &mut self,
        particle_id: u64,
        parent_id: Option<u64>,
        generation: u32,
        particle_type: ParticleType,
        position: [f64; 3],
        direction: [f64; 3],
        energy: f64,
        weight: f64,
        cell_id: Option<u32>,
    );

    /// Record a collision event
    fn record_collision(
        &mut self,
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
        position: [f64; 3],
        direction: [f64; 3],
        energy: f64,
        weight: f64,
        cell_id: Option<u32>,
    );

    /// Record a termination event (absorption or leak)
    fn record_termination(
        &mut self,
        position: [f64; 3],
        direction: [f64; 3],
        energy: f64,
        weight: f64,
        cell_id: Option<u32>,
        event_type: TrackEventType,
    );

    /// Next particle ID within the current history (0 for its first particle)
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
        _: u64,
        _: Option<u64>,
        _: u32,
        _: ParticleType,
        _: [f64; 3],
        _: [f64; 3],
        _: f64,
        _: f64,
        _: Option<u32>,
    ) {
    }

    #[inline(always)]
    fn record_collision(
        &mut self,
        _: [f64; 3],
        _: [f64; 3],
        _: f64,
        _: f64,
        _: f64,
        _: Option<u32>,
        _: i32,
        _: &str,
        _: Option<&str>,
        _: Option<&str>,
    ) {
    }

    #[inline(always)]
    fn record_surface_crossing(
        &mut self,
        _: [f64; 3],
        _: [f64; 3],
        _: f64,
        _: f64,
        _: Option<u32>,
    ) {
    }

    #[inline(always)]
    fn record_termination(
        &mut self,
        _: [f64; 3],
        _: [f64; 3],
        _: f64,
        _: f64,
        _: Option<u32>,
        _: TrackEventType,
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
    /// Finished tracks
    tracks: Vec<ParticleTrack>,
    /// Track of the particle currently being transported
    current_track: Option<ParticleTrack>,
    /// Current batch number
    current_batch: usize,
    /// Current global history index
    current_history: usize,
    /// Next particle ID within the current history
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

    /// Move the current track, if it has events, into the finished tracks.
    fn flush(&mut self) {
        if let Some(track) = self.current_track.take()
            && !track.events.is_empty()
        {
            self.tracks.push(track);
        }
    }

    /// Append an event to the current particle's track, returning it so the
    /// caller can fill in event-specific fields.
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        event_type: TrackEventType,
        position: [f64; 3],
        direction: [f64; 3],
        energy_in: f64,
        energy_out: f64,
        weight: f64,
        cell_id: Option<u32>,
    ) -> Option<&mut TrackEvent> {
        if !self.active {
            return None;
        }
        let track = self.current_track.as_mut()?;
        track.events.push(TrackEvent {
            particle_id: track.particle_id,
            parent_id: track.parent_id,
            generation: track.generation,
            particle_type: track.particle_type,
            batch: self.current_batch,
            history: self.current_history,
            event_type,
            position,
            direction,
            energy_in,
            energy_out,
            weight,
            cell_id,
            reaction_mt: None,
            nuclide: None,
            birth_reaction: None,
            distribution: None,
            energy_dist: None,
        });
        track.events.last_mut()
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
        self.flush();
        self.active = self.selection.contains(history as u64);
        self.current_batch = batch;
        self.current_history = history;
        self.next_id = 0;
    }

    fn finish_history(&mut self) {
        self.flush();
    }

    fn record_birth(
        &mut self,
        particle_id: u64,
        parent_id: Option<u64>,
        generation: u32,
        particle_type: ParticleType,
        position: [f64; 3],
        direction: [f64; 3],
        energy: f64,
        weight: f64,
        cell_id: Option<u32>,
    ) {
        if !self.active {
            return;
        }
        self.flush();
        self.current_track = Some(ParticleTrack {
            events: Vec::new(),
            particle_id,
            parent_id,
            generation,
            particle_type,
            history: self.current_history,
        });
        let (event_type, reaction) = if generation == 0 {
            (TrackEventType::SourceBirth, "source")
        } else {
            (TrackEventType::SecondaryBirth, "secondary")
        };
        if let Some(e) = self.push(
            event_type, position, direction, energy, energy, weight, cell_id,
        ) {
            e.birth_reaction = Some(reaction.to_string());
        }
    }

    fn record_collision(
        &mut self,
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
        let kind = TrackEventType::Collision;
        if let Some(e) = self.push(
            kind, position, direction, energy_in, energy_out, weight, cell_id,
        ) {
            e.reaction_mt = Some(mt);
            e.nuclide = Some(nuclide.to_string());
            e.distribution = distribution.map(str::to_string);
            e.energy_dist = energy_dist.map(str::to_string);
        }
    }

    fn record_surface_crossing(
        &mut self,
        position: [f64; 3],
        direction: [f64; 3],
        energy: f64,
        weight: f64,
        cell_id: Option<u32>,
    ) {
        let kind = TrackEventType::SurfaceCrossing;
        self.push(kind, position, direction, energy, energy, weight, cell_id);
    }

    fn record_termination(
        &mut self,
        position: [f64; 3],
        direction: [f64; 3],
        energy: f64,
        weight: f64,
        cell_id: Option<u32>,
        event_type: TrackEventType,
    ) {
        self.push(
            event_type, position, direction, energy, energy, weight, cell_id,
        );
    }

    fn next_particle_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id - 1
    }

    fn should_track(&self) -> bool {
        self.active
    }

    fn into_storage(mut self) -> TrackStorage {
        self.flush();
        TrackStorage {
            tracks: self.tracks,
        }
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const O: [f64; 3] = [0.0, 0.0, 0.0];
    const Z: [f64; 3] = [0.0, 0.0, 1.0];

    fn birth<T: Tracker>(
        t: &mut T,
        parent: Option<(u64, u32)>,
        particle_type: ParticleType,
    ) -> u64 {
        let id = t.next_particle_id();
        let generation = parent.map_or(0, |(_, g)| g + 1);
        t.record_birth(
            id,
            parent.map(|p| p.0),
            generation,
            particle_type,
            O,
            Z,
            1e6,
            1.0,
            Some(1),
        );
        id
    }

    #[test]
    fn test_noop_tracker() {
        let mut tracker = NoOpTracker::new();
        assert!(!tracker.should_track());
        tracker.start_history(0, 0);
        birth(&mut tracker, None, ParticleType::Neutron);
        tracker.finish_history();
        assert!(tracker.into_storage().tracks.is_empty());
    }

    #[test]
    fn test_real_tracker() {
        let mut tracker = RealTracker::new(HistorySelection::All);
        tracker.start_history(0, 0);
        assert!(tracker.should_track());
        assert_eq!(birth(&mut tracker, None, ParticleType::Neutron), 0);
        tracker.record_collision(
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
    fn test_each_particle_gets_its_own_track_with_type_and_lineage() {
        let mut tracker = RealTracker::new(HistorySelection::All);
        tracker.start_history(0, 7);
        let n = birth(&mut tracker, None, ParticleType::Neutron);
        tracker.record_surface_crossing(Z, Z, 1e6, 1.0, Some(2));
        let p = birth(&mut tracker, Some((n, 0)), ParticleType::Photon);
        tracker.record_termination(Z, Z, 1e5, 1.0, Some(2), TrackEventType::Leak);
        tracker.finish_history();

        let tracks = tracker.into_storage().tracks;
        assert_eq!(tracks.len(), 2);
        let (t0, t1) = (&tracks[0], &tracks[1]);
        assert_eq!((t0.particle_id, t0.parent_id, t0.generation), (0, None, 0));
        assert_eq!(
            (t1.particle_id, t1.parent_id, t1.generation),
            (p, Some(0), 1)
        );
        assert_eq!(
            (t0.particle_type, t1.particle_type),
            (ParticleType::Neutron, ParticleType::Photon)
        );
        assert_eq!((t0.history, t1.history), (7, 7));
        assert_eq!(t1.events[0].event_type, TrackEventType::SecondaryBirth);
        assert!(t1
            .events
            .iter()
            .all(|e| e.particle_type == ParticleType::Photon));
        assert_eq!(t1.events[1].event_type, TrackEventType::Leak);

        // Polylines: one line per track, offsets bracket each line's points.
        let mut storage = TrackStorage {
            tracks: tracks.clone(),
        };
        storage.sort();
        let lines = storage.polylines();
        assert_eq!(lines.offsets, vec![0, 2, 4]);
        assert_eq!(lines.particle_type, vec![0, 1]);
        assert_eq!(lines.parent_id, vec![-1, 0]);
        assert_eq!(lines.event_type[3], TrackEventType::Leak as u32);
        assert!(TrackEventType::ALL
            .iter()
            .enumerate()
            .all(|(i, t)| *t as usize == i));
        assert_eq!(lines.cell_id, vec![1, 2, 1, 2]);

        // IDs restart at 0 every history.
        let mut tracker = RealTracker::new(HistorySelection::All);
        tracker.start_history(0, 0);
        birth(&mut tracker, None, ParticleType::Neutron);
        tracker.start_history(0, 1);
        assert_eq!(tracker.next_particle_id(), 0);
    }

    #[test]
    fn test_first_n_selection() {
        // first(2) captures global indices 0 and 1 only.
        assert_eq!(
            captured_histories(HistorySelection::first(2), 5),
            vec![0, 1]
        );
    }

    /// Run histories 0..n through a tracker and return the captured global
    /// history indices (in capture order).
    fn captured_histories(selection: HistorySelection, n: usize) -> Vec<usize> {
        let mut tracker = RealTracker::new(selection);
        for i in 0..n {
            tracker.start_history(0, i);
            birth(&mut tracker, None, ParticleType::Neutron);
            tracker.finish_history();
        }
        tracker
            .into_storage()
            .tracks
            .iter()
            .map(|t| t.history)
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
        let range = |start, stop, step| HistorySelection::Range { start, stop, step };
        assert_eq!(
            captured_histories(range(20, 30, 1), 50),
            (20..30).collect::<Vec<_>>()
        );
        assert_eq!(
            captured_histories(range(0, 50, 10), 50),
            vec![0, 10, 20, 30, 40]
        );
        assert_eq!(
            captured_histories(
                HistorySelection::Set([20u64, 30, 45].into_iter().collect()),
                50
            ),
            vec![20, 30, 45]
        );
    }
}
