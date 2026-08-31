// Particle banking system
//
// Handles secondary particle queuing from multi-neutron reactions (n,2n), (n,3n), etc.
// May extend to eigenvalue/criticality work if that is added later.

use yamc_particle::particle::Particle;
use yamc_rng::secondary_seed;

/// Default capacity for the particle bank, matching the existing safety limit
/// (`MAX_PARTICLES_PER_HISTORY` in model.rs).
pub const DEFAULT_CAPACITY: usize = 1000;

/// Per-history bank of secondary particles produced during transport
/// (e.g. from (n,2n)/(n,3n) and secondary-photon production).
///
/// Internally a `Vec` used as a stack (LIFO), the same discipline OpenMC's
/// per-particle secondary bank uses (`Particle::secondary_bank`, drained with
/// `back()` / `pop_back()`).
///
/// # Why the order does not matter (issue #111)
///
/// Every banked particle carries the 32-bit seed of its OWN collision stream,
/// derived at BANK time from `(the seed of the walk that banked it, its ordinal
/// among that walk's secondaries)` via
/// [`secondary_seed`](yamc_rng::secondary_seed). The transport loop
/// re-seeds its PCG from that value when the particle is popped, instead of
/// letting the secondary continue whatever state the parent happened to leave
/// behind. What a secondary samples is therefore a function of its position in
/// the emission tree alone, not of when it was scheduled, so this stack order
/// and the GPU kernel's in-thread FIFO produce the same physics.
pub struct ParticleBank {
    /// Stack of particles to be transported (primary + secondaries)
    stack: Vec<Particle>,
    /// Parallel to [`stack`](Self::stack): the identity-derived collision seed
    /// of each banked particle. Kept beside the particles rather than on
    /// `Particle` so the hot, widely-cloned particle struct stays unchanged.
    seeds: Vec<u32>,
    /// Seed of the walk currently being transported, i.e. of the particle the
    /// last [`pop_particle`](Self::pop_particle) returned. Parent half of the
    /// key every [`bank_secondary`](Self::bank_secondary) derives from.
    walk_seed: u32,
    /// How many secondaries the current walk has banked, i.e. the ordinal the
    /// next [`bank_secondary`](Self::bank_secondary) will use. Reset on every
    /// pop, so each walk numbers its own secondaries from 0.
    walk_secondaries: u32,
    /// Cumulative count of weight-window split copies created during the
    /// current history. Used to bound splitting to a per-history budget so a
    /// deep, heavily-splitting history cannot explode the population; reset by
    /// [`ParticleBank::clear`] at the start of each history.
    ww_splits: usize,
    /// Cumulative count of fission neutrons produced during the current
    /// history, across every generation of its chain. Unlike
    /// [`ww_splits`](Self::ww_splits) it bounds nothing: the guard against a
    /// supercritical geometry being transported forever is on the LIVE bank
    /// depth (issue #348), and this is the diagnostic that names the cause when
    /// that guard, or the drain ceiling behind it, fires. Reset by
    /// [`ParticleBank::clear`].
    fission_progeny: usize,
}

/// The push / pop / seed accessors are `#[inline]`: they sit on the
/// per-walk path of the transport loop in another crate, and without LTO a
/// cross-crate call to them is not inlined. Once they carried the
/// secondary-seed bookkeeping as well as the `Vec` op, that call overhead
/// measured as ~3.5% of CPU transport throughput on an (n,2n)-heavy model
/// (Li6 core + Be9 shell at 14 MeV); with `#[inline]` the same model is back at
/// parity with the pre-#111 bank.
impl ParticleBank {
    /// Create a new empty particle bank
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Create a particle bank with an initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        ParticleBank {
            stack: Vec::with_capacity(capacity),
            seeds: Vec::with_capacity(capacity),
            walk_seed: 0,
            walk_secondaries: 0,
            ww_splits: 0,
            fission_progeny: 0,
        }
    }

    /// Weight-window split copies created so far this history.
    pub fn ww_splits(&self) -> usize {
        self.ww_splits
    }

    /// Record that `n` weight-window split copies were created this history.
    pub fn record_ww_splits(&mut self, n: usize) {
        self.ww_splits += n;
    }

    /// Record that a fission event produced `n` neutrons this history, and
    /// return the history's running total across every generation of its chain.
    ///
    /// Counts the neutrons the event SAMPLED, not the ones that ended up on the
    /// stack: the analog fission arm continues the walk as the first of them
    /// rather than banking it, and that neutron multiplies exactly like its
    /// siblings.
    #[inline]
    pub fn record_fission_progeny(&mut self, n: usize) -> usize {
        self.fission_progeny += n;
        self.fission_progeny
    }

    /// Fission neutrons produced so far this history.
    pub fn fission_progeny(&self) -> usize {
        self.fission_progeny
    }

    /// Add a primary particle to the bank, on the history's own collision
    /// stream. `seed` is the source particle's identity seed, i.e.
    /// `yamc_rng::history_seed(base_seed, global_index)`; it
    /// roots the whole history's secondary-seed tree.
    ///
    /// This also opens the source particle's walk, so a caller that banks
    /// secondaries without popping first (the photon-production unit tests do)
    /// still keys them on the source seed rather than on nothing.
    #[inline]
    pub fn add_source_particle(&mut self, particle: Particle, seed: u32) {
        self.stack.push(particle);
        self.seeds.push(seed);
        self.walk_seed = seed;
        self.walk_secondaries = 0;
    }

    /// Bank a secondary particle from a reaction (e.g., from n,2n or n,3n).
    ///
    /// The secondary's collision stream is derived HERE, from the banking
    /// walk's seed and the secondary's ordinal within that walk, so it does not
    /// depend on when the secondary is later popped (issue #111).
    #[inline]
    pub fn bank_secondary(&mut self, particle: Particle) {
        let seed = secondary_seed(self.walk_seed, self.walk_secondaries);
        self.walk_secondaries += 1;
        self.stack.push(particle);
        self.seeds.push(seed);
    }

    /// Get the next particle from the bank for transport. Returns `None` if the
    /// bank is empty.
    ///
    /// This BEGINS A NEW WALK: the popped particle's own collision seed becomes
    /// [`walk_seed`](Self::walk_seed) (which the caller re-seeds its PCG from),
    /// and secondaries banked from now on are keyed on it and numbered from 0.
    ///
    /// The seed is read back through [`walk_seed`](Self::walk_seed) rather than
    /// returned alongside the particle: the bank already owns that state (it is
    /// the key `bank_secondary` derives from), and keeping this signature avoids
    /// moving the ~140-byte `Particle` through a tuple at every call site,
    /// including the photon-production ones that do not care about the seed.
    #[inline]
    pub fn pop_particle(&mut self) -> Option<Particle> {
        let particle = self.stack.pop()?;
        self.walk_seed = self
            .seeds
            .pop()
            .expect("bank seed stack must stay parallel to the particle stack");
        self.walk_secondaries = 0;
        Some(particle)
    }

    /// Collision seed of the walk the last [`pop_particle`](Self::pop_particle)
    /// started, i.e. the 32-bit value that particle's PCG must be seeded from
    /// (issue #111). Before any pop this is the seed the source particle was
    /// added with, or 0 for a bank used without one.
    #[inline]
    pub fn walk_seed(&self) -> u32 {
        self.walk_seed
    }

    /// Check if the bank is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Get the number of particles in the bank
    #[inline]
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    /// Clear all particles from the bank and reset the per-history split and
    /// fission-progeny counts and the secondary-seed bookkeeping.
    pub fn clear(&mut self) {
        self.stack.clear();
        self.seeds.clear();
        self.walk_seed = 0;
        self.walk_secondaries = 0;
        self.ww_splits = 0;
        self.fission_progeny = 0;
    }

    /// Reserve capacity for additional particles
    pub fn reserve(&mut self, additional: usize) {
        self.stack.reserve(additional);
        self.seeds.reserve(additional);
    }
}

impl Default for ParticleBank {
    fn default() -> Self {
        Self::new()
    }
}
