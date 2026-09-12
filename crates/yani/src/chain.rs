/// Transmutation chain parser and cache.
///
/// Parses transmutation chain arrow directories and caches parsed chains
/// to avoid re-parsing on repeated transmute() calls.
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::sync::{Arc, RwLock};

use once_cell::sync::Lazy;

type ChainMap = HashMap<String, ChainNuclide>;
type ChainCache = RwLock<HashMap<String, Arc<ChainMap>>>;

/// A reaction or decay in the transmutation chain
#[derive(Clone, Debug)]
pub struct ChainReaction {
    /// Reaction type name, e.g. "(n,gamma)", "(n,fission)", "beta-", "alpha"
    pub kind: String,
    /// Target nuclide produced (if any)
    pub target: Option<String>,
    /// Branching ratio for this channel
    pub branching: f64,
    /// Q value in eV, for neutron-induced reactions.
    ///
    /// `None` on a decay mode: `decay/decay_modes.arrow` declares no Q column,
    /// and the same type carries both. It is `Some` for anything read from
    /// `reactions/reactions.arrow`, whose schema declares Q non-nullable.
    ///
    /// Carried rather than dropped because the writer has to put it back.
    /// Without it, loading a chain and re-exporting it silently returned a
    /// `reactions/reactions.arrow` with no Q column at all.
    pub q_value: Option<f64>,
}

/// The physical quantity tabulated in an isomeric-branching curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BranchQuantity {
    /// Multiplicity / branching fraction to the final state (MF=9); values are
    /// dimensionless and sum to ~1 across the final states of a reaction.
    Yield,
    /// Partial cross section to the final state (MF=10) in barns.
    CrossSection,
}

/// One verbatim, energy-dependent isomeric-branching curve for a single
/// (parent, reaction, final-state) triple, as stored in the `branching/`
/// subsection. The overlay folds these against the transport spectrum at
/// rate-compute time to obtain flux-weighted branching fractions.
///
/// # The branching subsection is a second source of cross sections
///
/// Worth stating plainly, because the name says "branch ratios" and the
/// setting sits beside `cross_section_data` as though the two were disjoint.
/// A [`BranchQuantity::CrossSection`] curve is a partial cross section in
/// barns, not a dimensionless fraction, and for `(n,n')` it is the *only*
/// thing the metastable production rate is computed from: `build_fold_refine`
/// folds it directly and grafts the result into the rates, without consulting
/// the parent's own evaluation at all.
///
/// Two consequences follow, and neither is obvious from the setting names.
///
/// * A run that names one library for `cross_section_data` and another for
///   `transmutation_branch_ratios` is taking cross sections from both. On the
///   published 2026-09-08 TENDL-2025 build, 39409 of the 40905 branching rows
///   are `CrossSection` rather than `Yield`, so this is the common case and
///   not a corner.
/// * Changing only the branching source therefore changes rates, not just how
///   an existing rate is split. On the FNS decay-heat benchmark, moving that
///   one setting from TENDL-2017 to ENDF/B-VIII.1 and holding the other four
///   fixed moves 78 of 132 experiments by more than a point and drops the
///   count agreeing within 10% from 64 to 38.
///
/// Anything reporting the provenance of a result should name the branching
/// source among its cross-section inputs, not only among its branching ones.
#[derive(Clone, Debug)]
pub struct BranchCurve {
    /// Final-state nuclide (GNDS name, e.g. "Ag110" ground or "Ag110_m1").
    pub target: String,
    /// Whether `values` are yields (fractions) or partial cross sections.
    pub quantity: BranchQuantity,
    /// Incident-neutron energy grid [eV], ascending.
    pub energy: Vec<f64>,
    /// Curve values on `energy`: fraction (Yield) or barns (CrossSection).
    pub values: Vec<f64>,
}

/// Isomeric-branching curves keyed by parent nuclide then reaction kind.
/// `branch_table[parent][kind]` is the list of per-final-state curves for that
/// reaction. Empty when no `branching/` subsection was supplied.
pub type BranchTable = HashMap<String, HashMap<String, Vec<BranchCurve>>>;

/// A parsed transmutation chain plus its optional isomeric-branching overlay.
///
/// The `chain` already has any `(n,n')` (self-inelastic) metastable-production
/// channels grafted on (topology); `branch` holds the verbatim energy-dependent
/// curves so the per-spectrum branching fractions can be folded at
/// rate-compute time. Both are `Arc` so the loaded result is cheap to clone and
/// cache.
#[derive(Clone)]
pub struct LoadedChain {
    pub chain: Arc<HashMap<String, ChainNuclide>>,
    pub branch: Arc<BranchTable>,
    /// Which optional subsections this chain was actually built from.
    pub parts: ChainParts,
}

/// Which optional subsections a chain carries.
///
/// The matrix builder needs this to tell a subsection that was left out from
/// one that is present but says nothing about a given nuclide. Both look like
/// an absence in the parsed chain -- no reactions on a nuclide, no fission
/// yields for it -- and only the first is a mistake worth stopping for. Plenty
/// of nuclides have a fission cross section and no evaluated yields, so
/// erroring on the second would reject libraries that are doing nothing wrong.
///
/// [`Default`] is everything present, which is what a chain parsed from a
/// directory of your own has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainParts {
    /// The `reactions/` subsection was loaded.
    pub reactions: bool,
    /// The `fission_yields/` subsection was loaded.
    pub fission_yields: bool,
}

impl Default for ChainParts {
    fn default() -> Self {
        Self {
            reactions: true,
            fission_yields: true,
        }
    }
}

/// Fission product yields at a single incident neutron energy.
#[derive(Clone, Debug)]
pub struct FissionYield {
    /// Incident neutron energy [eV]
    pub energy: f64,
    /// Product nuclide names and their yields (fractional per fission)
    pub products: Vec<(String, f64)>,
}

/// Complete fission yield data for a nuclide (may have multiple energies).
///
/// `yields` is kept sorted ascending by energy. Construct through
/// [`FissionYieldSet::new`] rather than the field directly: the spectrum fold
/// indexes the tabulated points positionally, so the order is part of the
/// contract, not an accident of the row order in the data file.
#[derive(Clone, Debug)]
pub struct FissionYieldSet {
    /// Yields at each energy point, ascending by energy.
    pub yields: Vec<FissionYield>,
}

impl FissionYieldSet {
    /// Build a set from yields in any order, sorting them ascending by energy.
    pub fn new(mut yields: Vec<FissionYield>) -> Self {
        yields.sort_by(|a, b| a.energy.total_cmp(&b.energy));
        Self { yields }
    }

    /// The tabulated incident energies [eV], ascending.
    ///
    /// Enough on its own to drive [`fission_yield_interp_weights`], so a hot
    /// path can fold against a nuclide's yields without holding its (~1000
    /// entry) product vectors.
    pub fn energies(&self) -> Vec<f64> {
        self.yields.iter().map(|y| y.energy).collect()
    }

    /// Bracketing tabulated points and their linear interpolation weights for
    /// an incident neutron `energy`. See [`fission_yield_interp_weights`].
    pub fn interp_weights(&self, energy: f64) -> Option<[(usize, f64); 2]> {
        // Same walk as the free function, without materialising the energies.
        let n = self.yields.len();
        if n == 0 {
            return None;
        }
        if energy <= self.yields[0].energy {
            return Some([(0, 1.0), (0, 0.0)]);
        }
        if energy >= self.yields[n - 1].energy {
            return Some([(n - 1, 1.0), (n - 1, 0.0)]);
        }
        let hi = self.yields.partition_point(|y| y.energy <= energy);
        Some(bracket(
            self.yields[hi - 1].energy,
            self.yields[hi].energy,
            hi,
            energy,
        ))
    }
}

/// Linear interpolation weights over `energies` (ascending) at `energy`.
///
/// Returns two `(index, weight)` pairs whose weights sum to one, so summing
/// them over a spectrum gives coefficients that are themselves a partition of
/// unity. Outside the tabulated range the interpolation is clamped flat,
/// putting all the weight on the nearest end point. Returns `None` for an empty
/// grid, which has nothing to interpolate.
pub fn fission_yield_interp_weights(energies: &[f64], energy: f64) -> Option<[(usize, f64); 2]> {
    let n = energies.len();
    if n == 0 {
        return None;
    }
    if energy <= energies[0] {
        return Some([(0, 1.0), (0, 0.0)]);
    }
    if energy >= energies[n - 1] {
        return Some([(n - 1, 1.0), (n - 1, 0.0)]);
    }
    // `hi` is the first point strictly above `energy`; the guards above leave
    // it in 1..n, so `hi - 1` brackets from below.
    let hi = energies.partition_point(|&e| e <= energy);
    Some(bracket(energies[hi - 1], energies[hi], hi, energy))
}

/// Split `energy` between the bracketing points `e_lo` (index `hi - 1`) and
/// `e_hi` (index `hi`). `e_hi > e_lo` because the grid is sorted and `energy`
/// lies strictly between them, so the division is safe.
fn bracket(e_lo: f64, e_hi: f64, hi: usize, energy: f64) -> [(usize, f64); 2] {
    let f = (energy - e_lo) / (e_hi - e_lo);
    [(hi - 1, 1.0 - f), (hi, f)]
}

/// Distribution data for a decay photon source.
#[derive(Clone, Debug)]
pub enum DecaySourceDistribution {
    /// Discrete line spectrum: each (energy, intensity) pair is a spectral line.
    Discrete {
        energies: Vec<f64>,
        intensities: Vec<f64>,
    },
}

/// A decay photon source associated with a nuclide.
///
/// In D1S chain files, nuclides may have source entries describing the decay
/// gamma spectrum emitted when the nuclide decays.
#[derive(Clone, Debug)]
pub struct DecaySource {
    /// Particle type emitted (e.g. "photon")
    pub particle: String,
    /// The energy distribution of emitted particles
    pub distribution: DecaySourceDistribution,
}

/// A nuclide in the transmutation chain
#[derive(Clone, Debug)]
pub struct ChainNuclide {
    /// Nuclide name, e.g. "U235", "Xe135"
    pub name: String,
    /// Half-life in seconds (None for stable nuclides)
    pub half_life: Option<f64>,
    /// The evaluation's standard deviation on the half-life, in seconds.
    ///
    /// `None` where none was published or the file predates the column,
    /// which is not the same as zero (issue #515).
    pub half_life_uncertainty: Option<f64>,
    /// Mean decay energy released per decay [eV].
    pub decay_energy: f64,
    /// The evaluation's standard deviation on `decay_energy`, in eV.
    ///
    /// `None` where none was published or the file predates the column. Decay
    /// heat is `activity * decay_energy`, so this scales the reported watts
    /// directly rather than diluting through a chain (issue #515).
    pub decay_energy_uncertainty: Option<f64>,
    /// Neutron-induced reactions
    pub reactions: Vec<ChainReaction>,
    /// Radioactive decay modes
    pub decays: Vec<ChainReaction>,
    /// Neutron fission yields (None if nuclide doesn't undergo fission or has no yield data)
    ///
    /// Behind an `Arc` because the whole `ChainNuclide` map is deep-cloned once
    /// per spectrum per `Material::transmute` call, by `refine_chain`, only so
    /// that a handful of branching fractions can be rewritten. The yields are
    /// never one of them, and they are most of the bytes: a fissile nuclide
    /// carries ~1000 product name `String`s per tabulated energy, so cloning
    /// 3820 entries copied millions of small allocations to change none of them
    /// (issue #576, finding 8). Most read sites reach through `Deref` unchanged.
    pub fission_yields: Option<Arc<FissionYieldSet>>,
    /// Decay photon sources (empty if nuclide has no decay gamma data)
    pub sources: Vec<DecaySource>,
}

/// Global cache for parsed chain files.
/// Maps directory path -> parsed chain data (Arc for cheap cloning).
static CHAIN_CACHE: Lazy<ChainCache> = Lazy::new(|| RwLock::new(HashMap::new()));

/// Cache for v2 subsection loads (chain + branching overlay), keyed by the
/// combined `(decay, reactions, fpy, branch)` path key.
static LOADED_CHAIN_CACHE: Lazy<RwLock<HashMap<String, LoadedChain>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Load a transmutation chain from a `.chain.arrow/` directory, using cache if available.
pub fn load_chain(path: &str) -> Result<Arc<HashMap<String, ChainNuclide>>, Box<dyn Error>> {
    {
        let cache = CHAIN_CACHE
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(chain) = cache.get(path) {
            return Ok(Arc::clone(chain));
        }
    }

    let mut cache = CHAIN_CACHE
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(chain) = cache.get(path) {
        return Ok(Arc::clone(chain));
    }

    if !Path::new(path).is_dir() {
        return Err(format!(
            "chain path '{path}' is not a directory; expected a `.chain.arrow/` layout"
        )
        .into());
    }
    let parsed = crate::chain_arrow::parse_chain_arrow(path)?;
    let chain = Arc::new(parsed);
    cache.insert(path.to_string(), Arc::clone(&chain));
    Ok(chain)
}

/// Load a transmutation chain from separate v2 subsection directories, plus an
/// optional `branching/` overlay directory, caching by the combined
/// `(decay, reactions, fission_yields, branch)` key so a mixed-library
/// combination caches distinctly from an all-one-library one.
///
/// When `branch_dir` is `Some`, the returned chain has `(n,n')` metastable
/// channels grafted on and the returned `branch` table carries the verbatim
/// energy-dependent curves for the rate-time fold; when `None`, `branch` is
/// empty and the chain is the plain three-part merge.
pub fn load_chain_parts(
    decay_dir: &str,
    reactions_dir: Option<&str>,
    fpy_dir: Option<&str>,
    branch_dir: Option<&str>,
) -> Result<LoadedChain, Box<dyn Error>> {
    // Combine the resolved paths into a single cache key (the unit separator
    // can't appear in a path). The empty branch slot keeps no-overlay loads
    // distinct from overlaid ones.
    let key = format!(
        "{decay_dir}\u{1f}{}\u{1f}{}\u{1f}{}",
        reactions_dir.unwrap_or(""),
        fpy_dir.unwrap_or(""),
        branch_dir.unwrap_or("")
    );
    {
        let cache = LOADED_CHAIN_CACHE
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(loaded) = cache.get(&key) {
            return Ok(loaded.clone());
        }
    }

    let mut cache = LOADED_CHAIN_CACHE
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(loaded) = cache.get(&key) {
        return Ok(loaded.clone());
    }

    let (chain, branch) = crate::chain_arrow::parse_chain_parts(
        Path::new(decay_dir),
        reactions_dir.map(Path::new),
        fpy_dir.map(Path::new),
        branch_dir.map(Path::new),
    )?;
    let loaded = LoadedChain {
        chain: Arc::new(chain),
        branch: Arc::new(branch),
        parts: ChainParts {
            reactions: reactions_dir.is_some(),
            fission_yields: fpy_dir.is_some(),
        },
    };
    cache.insert(key, loaded.clone());
    Ok(loaded)
}

/// Reduce a chain to only nuclides reachable from `initial_nuclides` within
/// `levels` reaction/decay steps. Returns a new chain map containing only
/// the reachable nuclides (including the initial ones).
///
/// Extract light-particle ejectile nuclide names from a reaction type string.
///
/// "(n,p)" produces H1, "(n,alpha)" produces He4, etc.
/// Decay modes like "alpha", "p", "beta-" also produce the corresponding nuclide.
fn ejectile_nuclides(kind: &str) -> Vec<&'static str> {
    let mut out = Vec::new();

    // Neutron reactions: "(n,XXX)" -- parse the product string after the comma
    if let Some(products) = kind.strip_prefix("(n,").and_then(|s| s.strip_suffix(')')) {
        // Walk the product string extracting particle tokens.
        // Match longest tokens first to avoid ambiguity (e.g. "gamma" vs "a").
        let mut s = products;
        while !s.is_empty() {
            // Multi-character tokens first (longest match)
            if s.starts_with("gamma") {
                s = &s[5..];
            } else if s.starts_with("fission") {
                s = &s[7..];
            } else if s.starts_with("3He") {
                out.push("He3");
                s = &s[3..];
            } else if s.starts_with("alpha") {
                out.push("He4");
                s = &s[5..];
            } else if s.starts_with('a') {
                out.push("He4");
                s = &s[1..];
            } else if s.starts_with('p') {
                out.push("H1");
                s = &s[1..];
            } else if s.starts_with('d') {
                out.push("H2");
                s = &s[1..];
            } else if s.starts_with('t') {
                out.push("H3");
                s = &s[1..];
            } else {
                // Digits (multipliers) and 'n' (neutrons) -- skip
                s = &s[1..];
            }
        }
    } else {
        // Decay modes
        match kind {
            "alpha" => out.push("He4"),
            "p" => out.push("H1"),
            "d" => out.push("H2"),
            "t" => out.push("H3"),
            _ => {}
        }
    }
    out
}

/// Every nuclide reachable from `initial_nuclides`, walked to saturation.
///
/// Follows the same four edge kinds [`reduce_chain`] does (reaction targets,
/// light-particle ejectiles, decay targets and fission-yield products), with no
/// depth limit, so the result is the transitive closure rather than a
/// truncation.
///
/// This is what a transmutation driver needs to decide whose cross sections to
/// load: the transmutation stepper only follows a reaction edge whose parent has
/// a loaded rate, and it walks a subset of these edges from a subset of these
/// seeds, so loading exactly this set cannot change any result. A depth-limited
/// walk can, which is why the depth argument lives on [`reduce_chain`] and not
/// here.
pub fn reachable_nuclides(
    chain: &HashMap<String, ChainNuclide>,
    initial_nuclides: &[&str],
) -> std::collections::HashSet<String> {
    walk(chain, initial_nuclides, usize::MAX)
}

/// The nuclides a transmutation can populate above `floor`, given how hard it
/// is actually driven.
///
/// [`reachable_nuclides`] answers a graph question, what is connected to what.
/// This answers the physical one: what can reach a density worth solving for.
/// Reachability alone is a poor guide, since from almost any seed it saturates
/// on one large strongly connected component, so an iron sphere and a water
/// cell walk to the same 1060 nuclides.
///
/// Each node carries an upper bound on the density it could attain over
/// `total_time`, and a node whose bound stays under `floor` is dropped along
/// with anything only it could feed. The bound never under-estimates:
///
/// * every incoming edge contributes before the test, and no partial sum
///   short-circuits it, so a nuclide fed weakly from many directions survives;
/// * a single edge transfers at most the parent's whole bound, because an atom
///   cannot transmute more than once;
/// * decay uses the exact `1 - exp(-ln2 t / T_half)` fraction, needing no
///   cross-section data at all.
///
/// `reaction_rate(parent, kind)` gives that channel's per-atom rate in 1/s, i.e.
/// the one-group cross section folded with the flux. Only parents already loaded
/// can be asked, which is the point: deciding whether to load a nuclide needs
/// its parents' cross sections, never its own.
///
/// The chain is cyclic, so this relaxes to a fixpoint rather than sweeping once.
/// Bounds rise monotonically and are capped, so it always terminates. The floor
/// is applied only to the converged answer, never during propagation, so a node
/// is judged on everything that reaches it however faintly.
///
/// # What a caller owes this function
///
/// Rate every parent, or the result is not a bound.
///
/// Returning zero for a nuclide the driver has not loaded does **not** give a
/// bound. Each such nuclide is individually under the floor, but a node fed by
/// enough of them clears it, and zeroed edges hide that. See
/// `many_sub_floor_parents_can_clear_the_floor_together`.
///
/// The obvious repair, growing the loaded set in rounds and finishing with a
/// verification sweep that rates the unloaded at a cross-section ceiling, is
/// sound: if the sweep adds nothing then even at a rate no nuclide can exceed,
/// nothing outside the set reaches the floor. It is also pointless. Measured on
/// the ENDF/B-VIII.1 chain from Fe56 (`examples/bound_prune.rs`), the ceiling
/// sweep terminates only once essentially the whole closure is loaded (1059 of
/// 1060 nodes), and then returns exactly the set that rating the closure
/// outright returns in one pass instead of thirteen. Growing the loaded set
/// buys nothing because proving a node absent needs its parents' rates just as
/// much as finding one present does.
///
/// So rate the closure. That costs the parents' cross sections, not a transport
/// solve per round: a rate is a cross section folded against a flux spectrum,
/// and `TransmutationTallies::bounding_reaction_rates` folds one for any
/// nuclide whose data is loaded. The only parents left at zero are then the
/// ones the run being pruned also rates at zero, which cannot react there
/// either.
pub fn populated_nuclides<F>(
    chain: &HashMap<String, ChainNuclide>,
    seeds: &HashMap<String, f64>,
    total_time: f64,
    floor: f64,
    reaction_rate: F,
) -> std::collections::HashSet<String>
where
    F: Fn(&str, &str) -> f64,
{
    use std::collections::HashSet;

    // A bound cannot exceed the material's own atom budget. Light ejectiles are
    // the only way one parent yields several products and no channel emits more
    // than a handful, so this ceiling keeps the fixpoint finite over the graph's
    // cycles without ever cutting a path a real solve could take.
    const MAX_YIELD_PER_ATOM: f64 = 8.0;
    // Relaxation stops once no bound moves by more than this, or after this many
    // sweeps. Both are numerical guards on a fixpoint, not accuracy settings:
    // the answer they converge to is the one `floor` is tested against.
    const CONVERGED: f64 = 1e-6;
    const MAX_SWEEPS: usize = 500;

    // Summed in name order. `ceiling` caps every bound below, and the final
    // filter is a threshold on those capped values, so a last-bit difference
    // here could move a nuclide across `floor` and change which products this
    // returns. That is the same reason `edges` is sorted further down: under
    // MPI every rank builds the layout from its own call, and two ranks
    // disagreeing about one nuclide leaves them with differently shaped buffers
    // to reduce.
    //
    // Unlike the other three sums fixed alongside this one, this is hardening
    // rather than a fix for anything observed: on the ENDF/B-8.1 SFR chain no
    // bound sits near enough to the floor for the ceiling's last bit to move
    // it, so a test written against it passed with and without this. It is here
    // because the margin is a property of the data rather than of the code.
    let mut seed_names: Vec<&String> = seeds.keys().collect();
    seed_names.sort();
    let total_seed: f64 = seed_names.iter().map(|name| seeds[*name]).sum();
    if total_seed <= 0.0 {
        return HashSet::new();
    }
    let ceiling = total_seed * MAX_YIELD_PER_ATOM;

    // Precompute the edge list as (parent, child, transferred fraction). The
    // fraction depends only on the parent and the channel, so it is built once
    // and reused every sweep.
    let mut edges: Vec<(&str, &str, f64)> = Vec::new();
    for (name, cn) in chain.iter() {
        for rx in &cn.reactions {
            let frac = (reaction_rate(name, &rx.kind) * total_time).min(1.0) * rx.branching;
            if frac <= 0.0 {
                continue;
            }
            if let Some(target) = &rx.target {
                if chain.contains_key(target) {
                    edges.push((name, target, frac));
                }
            }
            for ejectile in ejectile_nuclides(&rx.kind) {
                if chain.contains_key(ejectile) {
                    edges.push((name, ejectile, frac));
                }
            }
        }

        // Decay needs no cross section: the half-life gives the exact fraction
        // of a parent that has decayed by `total_time`.
        let decayed = match cn.half_life {
            Some(hl) if hl > 0.0 => 1.0 - (-std::f64::consts::LN_2 * total_time / hl).exp(),
            _ => 0.0,
        };
        if decayed > 0.0 {
            for dk in &cn.decays {
                let frac = decayed * dk.branching;
                if frac <= 0.0 {
                    continue;
                }
                if let Some(target) = &dk.target {
                    if chain.contains_key(target) {
                        edges.push((name, target, frac));
                    }
                }
                for ejectile in ejectile_nuclides(&dk.kind) {
                    if chain.contains_key(ejectile) {
                        edges.push((name, ejectile, frac));
                    }
                }
            }
        }

        // Fission opens a thousand channels at once, each with its own yield.
        if let Some(fy_set) = &cn.fission_yields {
            let fission_frac = (reaction_rate(name, "fission") * total_time).min(1.0);
            if fission_frac > 0.0 {
                for fy in &fy_set.yields {
                    for (product, yield_) in &fy.products {
                        if *yield_ > 0.0 && chain.contains_key(product) {
                            edges.push((name, product, fission_frac * yield_));
                        }
                    }
                }
            }
        }
    }

    // Chain iteration order is a `HashMap`'s, which Rust seeds per process, so
    // the same input would otherwise sum each node's incoming edges in a
    // different order on every run and land a last-bit apart. A driver turns
    // this answer into a tally layout, and under MPI every rank builds that
    // layout independently from its own call, so one nuclide sitting a bit
    // either side of the floor on one rank would leave the ranks with
    // differently shaped buffers to reduce. Sort once and the sum is fixed.
    edges.sort_unstable_by(|a, b| a.0.cmp(b.0).then(a.1.cmp(b.1)));

    // Relax b_j = seed_j + sum_i b_i * frac_ij to a fixpoint (a Neumann series,
    // kept finite by the ceiling).
    let mut bounds: HashMap<&str, f64> = HashMap::new();
    for (name, &d) in seeds {
        if let Some((k, _)) = chain.get_key_value(name) {
            bounds.insert(k.as_str(), d.min(ceiling));
        }
    }
    for _ in 0..MAX_SWEEPS {
        let mut accum: HashMap<&str, f64> = HashMap::new();
        for (parent, child, frac) in &edges {
            if let Some(&b) = bounds.get(parent) {
                if b > 0.0 {
                    *accum.entry(child).or_insert(0.0) += b * frac;
                }
            }
        }
        for (name, &d) in seeds {
            if let Some((k, _)) = chain.get_key_value(name) {
                *accum.entry(k.as_str()).or_insert(0.0) += d;
            }
        }

        let mut moved: f64 = 0.0;
        for (name, value) in accum {
            let capped = value.min(ceiling);
            let previous = bounds.get(name).copied().unwrap_or(0.0);
            if capped > previous {
                moved = moved.max((capped - previous) / capped.max(f64::MIN_POSITIVE));
                bounds.insert(name, capped);
            }
        }
        if moved < CONVERGED {
            break;
        }
    }

    bounds
        .into_iter()
        .filter(|&(_, b)| b >= floor)
        .map(|(name, _)| name.to_string())
        .collect()
}

/// Shared BFS behind [`reachable_nuclides`] and [`reduce_chain`], so the two
/// can never disagree about what an edge is.
fn walk(
    chain: &HashMap<String, ChainNuclide>,
    initial_nuclides: &[&str],
    levels: usize,
) -> std::collections::HashSet<String> {
    use std::collections::HashSet;

    // Collect all reachable nuclide names via BFS
    let mut visited: HashSet<String> = initial_nuclides.iter().map(|s| s.to_string()).collect();
    let mut frontier: Vec<String> = initial_nuclides.iter().map(|s| s.to_string()).collect();

    // `visited.insert(x.clone())` allocates whether or not the entry is new,
    // and on a real chain the overwhelming majority of visits are repeats: a
    // fissile network walks every product of every yield vector of every
    // fissionable parent, which came to ~300k `String` allocations per
    // `Material::transmute` call, all but a few thousand of them freed
    // immediately (issue #576, finding 8). Testing membership first costs one
    // extra hash lookup on the rare miss and saves the allocation on every hit.
    fn discover(name: &str, visited: &mut HashSet<String>, next_frontier: &mut Vec<String>) {
        if !visited.contains(name) {
            visited.insert(name.to_string());
            next_frontier.push(name.to_string());
        }
    }

    for _ in 0..levels {
        let mut next_frontier: Vec<String> = Vec::new();
        for name in &frontier {
            if let Some(nuclide) = chain.get(name) {
                // Walk reaction targets and extract light-particle ejectiles
                for rx in &nuclide.reactions {
                    if let Some(target) = &rx.target {
                        discover(target, &mut visited, &mut next_frontier);
                    }
                    // Add light-particle ejectiles (H1, H2, H3, He3, He4)
                    // that exist in the chain
                    for ejectile in ejectile_nuclides(&rx.kind) {
                        if chain.contains_key(ejectile) {
                            discover(ejectile, &mut visited, &mut next_frontier);
                        }
                    }
                }
                // Walk decay targets (no ejectile extraction for decay modes)
                for decay in &nuclide.decays {
                    if let Some(target) = &decay.target {
                        discover(target, &mut visited, &mut next_frontier);
                    }
                }
                // Walk fission yield products.
                // Important for uranium impurities in fusion reactor models.
                if let Some(ref fy_set) = nuclide.fission_yields {
                    for fy in &fy_set.yields {
                        for (product, _) in &fy.products {
                            if chain.contains_key(product) {
                                discover(product, &mut visited, &mut next_frontier);
                            }
                        }
                    }
                }
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }

    visited
}

pub fn reduce_chain(
    chain: &HashMap<String, ChainNuclide>,
    initial_nuclides: &[&str],
    levels: usize,
) -> HashMap<String, ChainNuclide> {
    let visited = walk(chain, initial_nuclides, levels);

    // Build reduced chain: include only visited nuclides, and prune
    // reaction/decay targets that point outside the reduced set.
    let mut reduced = HashMap::new();
    for name in &visited {
        if let Some(nuclide) = chain.get(name) {
            let mut nuc = nuclide.clone();
            // Keep reactions/decays but clear targets that point outside the reduced set
            for rx in nuc.reactions.iter_mut().chain(nuc.decays.iter_mut()) {
                if let Some(ref target) = rx.target {
                    if !visited.contains(target) {
                        rx.target = None;
                    }
                }
            }
            reduced.insert(name.clone(), nuc);
        }
    }
    reduced
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Ejectile nuclide extraction tests
    // =========================================================================

    #[test]
    fn test_ejectile_nuclides_basic() {
        assert_eq!(ejectile_nuclides("(n,p)"), vec!["H1"]);
        assert_eq!(ejectile_nuclides("(n,d)"), vec!["H2"]);
        assert_eq!(ejectile_nuclides("(n,t)"), vec!["H3"]);
        assert_eq!(ejectile_nuclides("(n,3He)"), vec!["He3"]);
        assert_eq!(ejectile_nuclides("(n,a)"), vec!["He4"]);
        assert_eq!(ejectile_nuclides("(n,alpha)"), vec!["He4"]);
    }

    #[test]
    fn test_ejectile_nuclides_no_ejectiles() {
        assert!(ejectile_nuclides("(n,gamma)").is_empty());
        assert!(ejectile_nuclides("(n,2n)").is_empty());
        assert!(ejectile_nuclides("(n,3n)").is_empty());
        assert!(ejectile_nuclides("(n,fission)").is_empty());
        assert!(ejectile_nuclides("beta-").is_empty());
        assert!(ejectile_nuclides("ec/beta+").is_empty());
    }

    #[test]
    fn test_ejectile_nuclides_compound() {
        assert_eq!(ejectile_nuclides("(n,np)"), vec!["H1"]);
        assert_eq!(ejectile_nuclides("(n,na)"), vec!["He4"]);
        assert_eq!(ejectile_nuclides("(n,nd)"), vec!["H2"]);
        assert_eq!(ejectile_nuclides("(n,nt)"), vec!["H3"]);
        assert_eq!(ejectile_nuclides("(n,2np)"), vec!["H1"]);
        assert_eq!(ejectile_nuclides("(n,2na)"), vec!["He4"]);
        assert_eq!(ejectile_nuclides("(n,n3He)"), vec!["He3"]);
        // Multiple light particles
        assert_eq!(ejectile_nuclides("(n,pa)"), vec!["H1", "He4"]);
        assert_eq!(ejectile_nuclides("(n,2p)"), vec!["H1"]);
        assert_eq!(ejectile_nuclides("(n,2a)"), vec!["He4"]);
        assert_eq!(ejectile_nuclides("(n,npa)"), vec!["H1", "He4"]);
        assert_eq!(ejectile_nuclides("(n,n2a)"), vec!["He4"]);
        assert_eq!(ejectile_nuclides("(n,da)"), vec!["H2", "He4"]);
    }

    #[test]
    fn test_ejectile_nuclides_decay_modes() {
        assert_eq!(ejectile_nuclides("alpha"), vec!["He4"]);
        assert_eq!(ejectile_nuclides("p"), vec!["H1"]);
        assert_eq!(ejectile_nuclides("d"), vec!["H2"]);
        assert_eq!(ejectile_nuclides("t"), vec!["H3"]);
    }

    // =========================================================================
    // reduce_chain tests -- build fixture chains in-memory
    // =========================================================================

    fn rx(kind: &str, target: Option<&str>, branching: f64) -> ChainReaction {
        ChainReaction {
            kind: kind.to_string(),
            target: target.map(|s| s.to_string()),
            branching,
            q_value: None,
        }
    }

    fn nuc(
        name: &str,
        half_life: Option<f64>,
        reactions: Vec<ChainReaction>,
        decays: Vec<ChainReaction>,
    ) -> ChainNuclide {
        ChainNuclide {
            name: name.to_string(),
            half_life,
            decay_energy: 0.0,
            reactions,
            decays,
            half_life_uncertainty: None,
            fission_yields: None,
            sources: Vec::new(),
            decay_energy_uncertainty: None,
        }
    }

    fn test_fixture_chain() -> HashMap<String, ChainNuclide> {
        let entries = vec![
            nuc("H1", None, vec![], vec![]),
            nuc("H2", None, vec![], vec![]),
            nuc("He4", None, vec![], vec![]),
            nuc(
                "Fe56",
                None,
                vec![
                    rx("(n,gamma)", Some("Fe57"), 1.0),
                    rx("(n,p)", Some("Mn56"), 1.0),
                ],
                vec![],
            ),
            nuc(
                "Fe57",
                None,
                vec![rx("(n,gamma)", Some("Fe58"), 1.0)],
                vec![],
            ),
            nuc("Fe58", None, vec![], vec![]),
            nuc(
                "Mn56",
                Some(9284.4),
                vec![],
                vec![rx("beta-", Some("Fe56"), 1.0)],
            ),
            nuc(
                "Ag108",
                Some(142.92),
                vec![
                    rx("(n,2n)", Some("Ag107"), 1.0),
                    rx("(n,gamma)", Some("Ag109"), 1.0),
                    rx("(n,p)", Some("Pd108"), 1.0),
                    rx("(n,a)", Some("Rh105"), 1.0),
                ],
                vec![
                    rx("ec/beta+", Some("Pd108"), 0.0285),
                    rx("beta-", Some("Cd108"), 0.9715),
                ],
            ),
            nuc("Ag107", None, vec![], vec![]),
            nuc("Ag109", None, vec![], vec![]),
            nuc("Pd108", None, vec![], vec![]),
            nuc("Rh105", None, vec![], vec![]),
            nuc("Cd108", None, vec![], vec![]),
            nuc(
                "Th231",
                Some(91872.0),
                vec![],
                vec![rx("beta-", Some("Pa231"), 1.0)],
            ),
            nuc("Pa231", None, vec![], vec![]),
            nuc("Xe135", None, vec![], vec![]),
            nuc("I135", None, vec![], vec![]),
            nuc("Cs137", None, vec![], vec![]),
            nuc("Sr90", None, vec![], vec![]),
        ];
        let mut chain: HashMap<String, ChainNuclide> =
            entries.into_iter().map(|n| (n.name.clone(), n)).collect();

        // U235 with fission yields to Xe135/I135/Cs137/Sr90
        let mut u235 = nuc(
            "U235",
            Some(2.22102e16),
            vec![
                rx("(n,gamma)", Some("U236"), 1.0),
                rx("(n,fission)", None, 1.0),
            ],
            vec![rx("alpha", Some("Th231"), 1.0)],
        );
        u235.fission_yields = Some(Arc::new(FissionYieldSet {
            yields: vec![FissionYield {
                energy: 0.0253,
                products: vec![
                    ("Xe135".into(), 0.065),
                    ("I135".into(), 0.063),
                    ("Cs137".into(), 0.062),
                    ("Sr90".into(), 0.058),
                ],
            }],
        }));
        chain.insert("U235".into(), u235);

        // U236 inherits U235 yields (simulated by direct copy)
        let u236_yields = chain.get("U235").unwrap().fission_yields.clone();
        let mut u236 = nuc("U236", None, vec![rx("(n,fission)", None, 1.0)], vec![]);
        u236.fission_yields = u236_yields;
        chain.insert("U236".into(), u236);

        chain
    }

    #[test]
    fn reachable_nuclides_saturates_and_is_a_superset_of_any_level() {
        // The closure must not depend on a depth, and must contain every
        // level-limited walk, since that is what makes loading it exactly
        // result-preserving.
        let chain = test_fixture_chain();
        let full = reachable_nuclides(&chain, &["Fe56"]);
        for level in [1usize, 2, 3, 5, 20] {
            let reduced = reduce_chain(&chain, &["Fe56"], level);
            for name in reduced.keys() {
                assert!(
                    full.contains(name),
                    "level {level} reached {name}, which the saturating closure missed"
                );
            }
        }
        // Saturating means a deeper walk adds nothing.
        assert_eq!(full, reachable_nuclides(&chain, &["Fe56"]));
    }

    /// Seeds at unit density, so a bound reads as a fraction of the material.
    fn fe56_seed() -> HashMap<String, f64> {
        [("Fe56".to_string(), 1.0)].into_iter().collect()
    }

    #[test]
    fn populated_is_always_within_the_reachable_closure() {
        // The bound decides how much of the closure to keep. It must never
        // invent a nuclide the graph cannot reach, at any drive.
        let chain = test_fixture_chain();
        let closure = reachable_nuclides(&chain, &["Fe56"]);
        for rate in [0.0, 1e-20, 1e-6, 1.0, 1e6] {
            let kept = populated_nuclides(&chain, &fe56_seed(), 3.15e7, 1e-30, |_, _| rate);
            for name in &kept {
                assert!(
                    closure.contains(name),
                    "rate {rate:e} kept {name}, which is not reachable at all"
                );
            }
        }
    }

    #[test]
    fn populated_grows_with_drive_and_saturates_at_the_closure() {
        // The whole point of replacing a hop count: depth has to track fluence.
        // Harder driving reaches further, and the closure is the ceiling.
        let chain = test_fixture_chain();
        let closure = reachable_nuclides(&chain, &["Fe56"]);
        let sizes: Vec<usize> = [1e-30, 1e-12, 1e-8, 1e-4, 1.0]
            .iter()
            .map(|&rate| populated_nuclides(&chain, &fe56_seed(), 3.15e7, 1e-30, |_, _| rate).len())
            .collect();
        for pair in sizes.windows(2) {
            assert!(
                pair[1] >= pair[0],
                "a harder-driven run reached fewer nuclides: {sizes:?}"
            );
        }
        let hardest = populated_nuclides(&chain, &fe56_seed(), 3.15e7, 1e-30, |_, _| 1.0);
        assert_eq!(
            hardest.len(),
            closure.len(),
            "saturating drive should reach the whole closure"
        );
    }

    #[test]
    fn populated_keeps_a_decay_daughter_with_no_cross_sections() {
        // Mn56 decays to Fe56 with a 2.6 h half-life. Decay needs no rate, so
        // the daughter has to survive even when every reaction rate is zero.
        let chain = test_fixture_chain();
        let seeds: HashMap<String, f64> = [("Mn56".to_string(), 1.0)].into_iter().collect();
        let kept = populated_nuclides(&chain, &seeds, 3.15e7, 1e-30, |_, _| 0.0);
        assert!(
            kept.contains("Fe56"),
            "decay daughter dropped though decay needs no cross section: {kept:?}"
        );
    }

    #[test]
    fn populated_drops_what_the_drive_cannot_reach() {
        // At a rate low enough that one hop lands under the floor, the two-hop
        // product must go, while the one-hop product stays. This is the pruning
        // the closure alone cannot do.
        let chain = test_fixture_chain();
        // One hop transfers 1e-20 of the seed, two hops 1e-40, floor is 1e-30.
        let kept = populated_nuclides(&chain, &fe56_seed(), 1.0, 1e-30, |_, kind| {
            if kind.starts_with("(n,") {
                1e-20
            } else {
                0.0
            }
        });
        assert!(kept.contains("Fe57"), "one hop should survive: {kept:?}");
        assert!(
            !kept.contains("Fe58"),
            "two hops land under the floor and should be dropped: {kept:?}"
        );
    }

    #[test]
    fn populated_sums_pathways_rather_than_taking_the_strongest() {
        // A nuclide fed weakly from many parents can clear the floor when no
        // single path would. Summing every incoming edge before testing is what
        // makes the bound safe, so check the sum actually happens.
        let mut chain = HashMap::new();
        chain.insert("Seed".to_string(), {
            let mut cn = nuc("Seed", None, vec![], vec![]);
            cn.reactions = (0..20)
                .map(|_| rx("(n,gamma)", Some("Sink"), 1.0))
                .collect();
            cn
        });
        chain.insert("Sink".to_string(), nuc("Sink", None, vec![], vec![]));
        let seeds: HashMap<String, f64> = [("Seed".to_string(), 1.0)].into_iter().collect();
        // Each of the 20 edges alone is under the floor; together they clear it.
        let kept = populated_nuclides(&chain, &seeds, 1.0, 1e-30, |_, _| 1e-31);
        assert!(
            kept.contains("Sink"),
            "twenty sub-floor pathways should add up to clear the floor: {kept:?}"
        );
    }

    #[test]
    fn many_sub_floor_parents_can_clear_the_floor_together() {
        // Why a driver may not rate its unloaded nuclides at zero.
        //
        // A seed feeds 200 intermediates, each landing at 1e-32, well under the
        // 1e-30 floor, so a round-based driver would not load any of them. Every
        // intermediate feeds the same sink. Their contributions add to 2e-30 and
        // the sink clears the floor, but only if the intermediates are rated.
        // Rate them at zero, as an unloaded nuclide would be, and the sink
        // vanishes: that is the caller contract on `populated_nuclides` failing,
        // and the ceiling sweep is what catches it.
        const N: usize = 200;
        let mut chain = HashMap::new();
        let mut seed = nuc("Seed", None, vec![], vec![]);
        seed.reactions = (0..N)
            .map(|i| rx("(n,gamma)", Some(&format!("Mid{i}")), 1.0))
            .collect();
        chain.insert("Seed".to_string(), seed);
        for i in 0..N {
            let mut mid = nuc(&format!("Mid{i}"), None, vec![], vec![]);
            mid.reactions = vec![rx("(n,gamma)", Some("Sink"), 1.0)];
            chain.insert(format!("Mid{i}"), mid);
        }
        chain.insert("Sink".to_string(), nuc("Sink", None, vec![], vec![]));

        let seeds: HashMap<String, f64> = [("Seed".to_string(), 1.0)].into_iter().collect();
        let floor = 1e-30;

        // Every intermediate really is under the floor on its own.
        let ceiling_sweep = populated_nuclides(&chain, &seeds, 1.0, floor, |parent, _| {
            if parent == "Seed" {
                1e-32
            } else {
                1.0 // the ceiling: no nuclide can react faster than every atom
            }
        });
        assert!(
            !ceiling_sweep.contains("Mid0"),
            "an intermediate at 1e-32 is under the floor and should not be kept"
        );
        assert!(
            ceiling_sweep.contains("Sink"),
            "200 parents at 1e-32 sum to 2e-30 and must clear a 1e-30 floor: {}",
            ceiling_sweep.len()
        );

        // Rating the unloaded intermediates at zero silently loses the sink.
        let zeroed = populated_nuclides(&chain, &seeds, 1.0, floor, |parent, _| {
            if parent == "Seed" {
                1e-32
            } else {
                0.0
            }
        });
        assert!(
            !zeroed.contains("Sink"),
            "zeroing unloaded parents should be what loses the sink, \
             otherwise this test is not demonstrating the hazard"
        );
    }

    #[test]
    fn test_reduce_chain_includes_ejectiles() {
        // Fe56 has (n,p) → Mn56 (ejectile: H1) and (n,gamma) → Fe57
        let chain = test_fixture_chain();
        let reduced = reduce_chain(&chain, &["Fe56"], 1);
        let names: std::collections::HashSet<&str> = reduced.keys().map(|s| s.as_str()).collect();

        assert!(names.contains("Fe56"), "Seed nuclide");
        assert!(names.contains("Fe57"), "(n,gamma) target");
        assert!(names.contains("Mn56"), "(n,p) target");
        assert!(names.contains("H1"), "(n,p) ejectile");
    }

    #[test]
    fn test_reduce_chain_alpha_decay_no_ejectile() {
        // U235 decays via alpha -> Th231. Ejectiles are NOT extracted
        // from decay modes, only from neutron reactions.
        let chain = test_fixture_chain();
        let reduced = reduce_chain(&chain, &["U235"], 1);
        let names: std::collections::HashSet<&str> = reduced.keys().map(|s| s.as_str()).collect();

        assert!(names.contains("U235"));
        assert!(names.contains("Th231"), "alpha decay target");
        assert!(names.contains("U236"), "(n,gamma) target");
        // He4 is NOT included from alpha decay (decay ejectiles not extracted)
        assert!(!names.contains("He4"), "decay ejectiles not extracted");
    }

    #[test]
    fn test_reduce_chain_ag108_ejectiles() {
        // Ag108 has (n,p)→Pd108, (n,a)→Rh105 -- ejectiles H1 and He4
        let chain = test_fixture_chain();
        let reduced = reduce_chain(&chain, &["Ag108"], 1);
        let names: std::collections::HashSet<&str> = reduced.keys().map(|s| s.as_str()).collect();

        assert!(names.contains("H1"), "(n,p) ejectile");
        assert!(names.contains("He4"), "(n,a) ejectile");
        assert!(names.contains("Pd108"), "(n,p) target / ec decay target");
        assert!(names.contains("Rh105"), "(n,a) target");
    }

    #[test]
    fn test_reduce_chain_fission_yields() {
        // U235 has fission yields with products Xe135, I135, Cs137, Sr90.
        let chain = test_fixture_chain();
        let reduced = reduce_chain(&chain, &["U235"], 1);
        let names: std::collections::HashSet<&str> = reduced.keys().map(|s| s.as_str()).collect();

        assert!(names.contains("U236"), "(n,gamma) target");
        assert!(names.contains("Th231"), "alpha decay target");

        // Fission yield products
        assert!(names.contains("Xe135"), "fission product");
        assert!(names.contains("I135"), "fission product");
        assert!(names.contains("Cs137"), "fission product");
        assert!(names.contains("Sr90"), "fission product");
    }

    // --- fission-yield energy interpolation (issue #379) ---

    fn fy_at(energy: f64) -> FissionYield {
        FissionYield {
            energy,
            products: vec![],
        }
    }

    #[test]
    fn fission_yield_set_sorts_on_construction() {
        // Arrow row order is not a contract, so the constructor imposes one.
        let set = FissionYieldSet::new(vec![fy_at(1.4e7), fy_at(0.0253), fy_at(5.0e5)]);
        assert_eq!(set.energies(), vec![0.0253, 5.0e5, 1.4e7]);
    }

    #[test]
    fn interp_weights_are_a_partition_of_unity() {
        let set = FissionYieldSet::new(vec![fy_at(0.0253), fy_at(5.0e5), fy_at(1.4e7)]);
        // Inside the range, below it, above it, and exactly on each point.
        for e in [1e-5, 0.0253, 1.0, 5.0e5, 1.0e6, 1.4e7, 2.0e7] {
            let w = set.interp_weights(e).expect("non-empty set");
            let total: f64 = w.iter().map(|(_, w)| w).sum();
            assert!(
                (total - 1.0).abs() < 1e-15,
                "weights at {e} eV sum to {total}, not 1"
            );
        }
    }

    #[test]
    fn interp_weights_clamp_flat_outside_the_tabulated_range() {
        let set = FissionYieldSet::new(vec![fy_at(0.0253), fy_at(5.0e5), fy_at(1.4e7)]);

        // Below the first point: all the weight on the first point.
        let below = set.interp_weights(1e-5).unwrap();
        assert_eq!(below[0], (0, 1.0));
        assert_eq!(below[1].1, 0.0);

        // Above the last: all the weight on the last point.
        let above = set.interp_weights(2.0e7).unwrap();
        assert_eq!(above[0], (2, 1.0));
        assert_eq!(above[1].1, 0.0);
    }

    #[test]
    fn interp_weights_reproduce_a_tabulated_point_exactly() {
        // The exactness property the fold rests on: a delta spectrum sitting on
        // a tabulated energy selects that energy's yield vector alone, so the
        // fold is bit-identical to reading that vector directly.
        let set = FissionYieldSet::new(vec![fy_at(0.0253), fy_at(5.0e5), fy_at(1.4e7)]);
        for (k, e) in [0.0253, 5.0e5, 1.4e7].iter().enumerate() {
            let w = set.interp_weights(*e).unwrap();
            let mut c = vec![0.0; 3];
            for (i, weight) in w {
                c[i] += weight;
            }
            let mut expected = vec![0.0; 3];
            expected[k] = 1.0;
            assert_eq!(c, expected, "delta at {e} eV must select point {k} alone");
        }
    }

    #[test]
    fn interp_weights_bisect_the_midpoint() {
        let set = FissionYieldSet::new(vec![fy_at(0.0), fy_at(100.0)]);
        let w = set.interp_weights(25.0).unwrap();
        assert_eq!(w, [(0, 0.75), (1, 0.25)]);
    }

    #[test]
    fn interp_weights_on_an_empty_set_are_undefined() {
        assert!(FissionYieldSet::new(vec![]).interp_weights(1.0).is_none());
        assert!(fission_yield_interp_weights(&[], 1.0).is_none());
    }

    #[test]
    fn free_interp_weights_match_the_method() {
        let set = FissionYieldSet::new(vec![fy_at(0.0253), fy_at(5.0e5), fy_at(1.4e7)]);
        let energies = set.energies();
        for e in [1e-5, 0.0253, 3.0, 5.0e5, 7.5e6, 1.4e7, 3.0e7] {
            assert_eq!(
                set.interp_weights(e),
                fission_yield_interp_weights(&energies, e),
                "hot-path form must agree with the method at {e} eV"
            );
        }
    }
}
