/// Transmutation matrix builder.
///
/// Constructs the transmutation matrix A for the Bateman equations from
/// chain data and reaction rates.
use std::collections::{BTreeMap, HashMap};

use crate::chain::{ChainNuclide, ChainParts, ChainReaction};
use crate::ReactionRates;

/// Per-nuclide spectrum weights over its tabulated fission-yield energies.
///
/// `nuclide -> [c_0, c_1, ...]`, one coefficient per entry of that nuclide's
/// [`FissionYieldSet::yields`](crate::chain::FissionYieldSet::yields), in the
/// same (ascending-energy) order. The effective yield vector is the combination
///
/// ```text
/// Y_eff(p) = Sum_k c_k * Y_k(p)
/// ```
///
/// The drivers own the spectrum and compute `c_k` by summing the linear
/// interpolation weights of [`FissionYieldSet::interp_weights`] over the
/// fission-rate distribution `R_g = sigma_f,g * phi_g`, normalized so the
/// coefficients sum to one. That is the exact commuted form of folding the
/// yield vectors group by group, since the interpolation is linear and so
/// commutes with the sum, and it costs `n_groups + 4 * n_products` instead of
/// `n_groups * n_products`.
pub type FissionYieldWeights = HashMap<String, Vec<f64>>;

/// A matrix in COO form: `(row, col, value)` triplets plus the dimension.
pub type MatrixTriplets = (Vec<(usize, usize, f64)>, usize);

/// Per-edge neutron-induced reaction rates for one material over one step:
/// `parent -> reaction kind -> [(target, rate [1/s])]`.
///
/// One entry per production edge the solve actually drove, which is what a
/// pathway analysis weights its routes by. `target` is `None` on a channel that
/// names no single product, as it is on
/// [`ChainReaction::target`](crate::chain::ChainReaction::target), which in
/// practice means fission.
pub type EdgeRates = HashMap<String, HashMap<String, Vec<(Option<String>, f64)>>>;

/// Split each nuclide's reaction rate over the edges it drives.
///
/// The rate a matrix entry carries is `rx.branching * rates[parent][kind]`
/// (see the reaction branch of [`accumulate_matrix`]), so that product is what
/// this returns, per `(parent, kind, target)`. It is the quantity the solve
/// computed and then discarded: the burnup matrix keeps the sum over edges into
/// each product, and nothing kept the edges.
///
/// Both transmutation methods reach this through the same two arguments, so
/// neither the shape nor the meaning depends on which one ran. `rates` come
/// from the transport tallies on the coupled path and from the multigroup fold
/// on the independent one, and `chain` is in both cases the chain the step was
/// solved with, isomeric-branching overlay included. Fission is one edge per
/// fission channel carrying the whole channel rate; its products come from the
/// yields rather than from an edge, so they are not enumerated here.
///
/// A parent with no rate contributes nothing, so a decay-only step gives an
/// empty map.
pub fn per_edge_rates(chain: &HashMap<String, ChainNuclide>, rates: &ReactionRates) -> EdgeRates {
    let mut out: EdgeRates = HashMap::new();
    for (parent, rate_map) in rates {
        let Some(nuc) = chain.get(parent.as_str()) else {
            continue;
        };
        for rx in &nuc.reactions {
            let Some(&rate) = rate_map.get(rx.kind.as_str()) else {
                continue;
            };
            if rate == 0.0 {
                continue;
            }
            // A channel with no named product (fission) is still an edge worth
            // reporting: it is where the rate went.
            let target = rx.target.clone();
            let edges = out
                .entry(parent.clone())
                .or_default()
                .entry(rx.kind.clone())
                .or_default();
            // Duplicate curves for one target sum rather than overwrite, as
            // they do in the branching overlay.
            match edges.iter_mut().find(|(t, _)| *t == target) {
                Some((_, r)) => *r += rx.branching * rate,
                None => edges.push((target, rx.branching * rate)),
            }
        }
    }
    out
}

/// Map decay types to the light particles they emit.
/// Returns (nuclide_name, count) pairs.
/// The target nuclide is handled separately - this only returns the emitted particles.
/// Maps each decay mode to the light particles it emits.
pub fn decay_particle_products(decay_type: &str) -> &'static [(&'static str, f64)] {
    match decay_type {
        // Alpha decay emits one He4 (the target is the residual nucleus)
        "alpha" => &[("He4", 1.0)],

        // Proton decay emits one H1
        "p" => &[("H1", 1.0)],
        "p,p" => &[("H1", 2.0)],

        // Delayed particle emissions (beta followed by particle)
        "beta-,alpha" => &[("He4", 1.0)],
        "beta-,2alpha" => &[("He4", 2.0)],
        "ec/beta+,alpha" => &[("He4", 1.0)],
        "ec/beta+,2alpha" => &[("He4", 2.0)],
        "ec/beta+,p" => &[("H1", 1.0)],
        "ec/beta+,2p" => &[("H1", 2.0)],

        // Beta decay modes without particle emission
        // (electrons/positrons don't affect nuclide inventory)
        "beta-" | "beta+" | "ec/beta+" | "ec" | "IT" | "SF" => &[],

        // Neutron emission (doesn't affect nuclide inventory for transmutation purposes)
        "n" | "beta-,n" | "beta-,n,n" | "beta-,2n" | "ec/beta+,n" => &[],

        // Unknown decay type - no particles
        _ => &[],
    }
}

/// Map reaction types to the light particles they emit.
/// Returns (nuclide_name, count) pairs.
/// Maps each neutron reaction type to the light particles it emits.
pub fn light_particle_products(reaction_type: &str) -> &'static [(&'static str, f64)] {
    match reaction_type {
        // Single particle emissions
        "(n,p)" => &[("H1", 1.0)],
        "(n,d)" => &[("H2", 1.0)],
        "(n,t)" => &[("H3", 1.0)],
        "(n,3He)" => &[("He3", 1.0)],
        "(n,a)" => &[("He4", 1.0)],

        // Double particle emissions
        "(n,2p)" => &[("H1", 2.0)],
        "(n,2a)" => &[("He4", 2.0)],
        "(n,np)" => &[("H1", 1.0)],
        "(n,nd)" => &[("H2", 1.0)],
        "(n,nt)" => &[("H3", 1.0)],
        "(n,na)" => &[("He4", 1.0)],
        "(n,pa)" => &[("H1", 1.0), ("He4", 1.0)],
        "(n,da)" => &[("H2", 1.0), ("He4", 1.0)],
        "(n,ta)" => &[("H3", 1.0), ("He4", 1.0)],
        "(n,pd)" => &[("H1", 1.0), ("H2", 1.0)],
        "(n,pt)" => &[("H1", 1.0), ("H3", 1.0)],
        "(n,dt)" => &[("H2", 1.0), ("H3", 1.0)],
        "(n,n3He)" => &[("He3", 1.0)],
        "(n,p3He)" => &[("H1", 1.0), ("He3", 1.0)],
        "(n,d3He)" => &[("H2", 1.0), ("He3", 1.0)],
        "(n,3Hea)" => &[("He3", 1.0), ("He4", 1.0)],

        // Triple particle emissions
        "(n,3a)" => &[("He4", 3.0)],
        "(n,n2a)" => &[("He4", 2.0)],
        "(n,n3a)" => &[("He4", 3.0)],
        "(n,2na)" => &[("He4", 1.0)],
        "(n,3na)" => &[("He4", 1.0)],
        "(n,2n2a)" => &[("He4", 2.0)],
        "(n,t2a)" => &[("H3", 1.0), ("He4", 2.0)],
        "(n,d2a)" => &[("H2", 1.0), ("He4", 2.0)],
        "(n,nd2a)" => &[("H2", 1.0), ("He4", 2.0)],
        "(n,nt2a)" => &[("H3", 1.0), ("He4", 2.0)],
        "(n,npa)" => &[("H1", 1.0), ("He4", 1.0)],
        "(n,nda)" => &[("H2", 1.0), ("He4", 1.0)],
        "(n,nta)" => &[("H3", 1.0), ("He4", 1.0)],
        "(n,2np)" => &[("H1", 1.0)],
        "(n,3np)" => &[("H1", 1.0)],
        "(n,n2p)" => &[("H1", 2.0)],
        "(n,2nd)" => &[("H2", 1.0)],
        "(n,2nt)" => &[("H3", 1.0)],
        "(n,npd)" => &[("H1", 1.0), ("H2", 1.0)],
        "(n,npt)" => &[("H1", 1.0), ("H3", 1.0)],
        "(n,ndt)" => &[("H2", 1.0), ("H3", 1.0)],
        "(n,np3He)" => &[("H1", 1.0), ("He3", 1.0)],
        "(n,nd3He)" => &[("H2", 1.0), ("He3", 1.0)],
        "(n,nt3He)" => &[("H3", 1.0), ("He3", 1.0)],
        "(n,3p)" => &[("H1", 3.0)],
        "(n,n3p)" => &[("H1", 3.0)],
        "(n,2n2p)" => &[("H1", 2.0)],
        "(n,2npa)" => &[("H1", 1.0), ("He4", 1.0)],

        // Higher multiplicity emissions
        "(n,4np)" => &[("H1", 1.0)],
        "(n,5np)" => &[("H1", 1.0)],
        "(n,6np)" => &[("H1", 1.0)],
        "(n,7np)" => &[("H1", 1.0)],
        "(n,3nd)" => &[("H2", 1.0)],
        "(n,4nd)" => &[("H2", 1.0)],
        "(n,5nd)" => &[("H2", 1.0)],
        "(n,6nd)" => &[("H2", 1.0)],
        "(n,3nt)" => &[("H3", 1.0)],
        "(n,4nt)" => &[("H3", 1.0)],
        "(n,5nt)" => &[("H3", 1.0)],
        "(n,6nt)" => &[("H3", 1.0)],
        "(n,4na)" => &[("He4", 1.0)],
        "(n,5na)" => &[("He4", 1.0)],
        "(n,6na)" => &[("He4", 1.0)],
        "(n,7na)" => &[("He4", 1.0)],
        "(n,2n3He)" => &[("He3", 1.0)],
        "(n,3n3He)" => &[("He3", 1.0)],
        "(n,4n3He)" => &[("He3", 1.0)],
        "(n,3n2p)" => &[("H1", 2.0)],
        "(n,3n2a)" => &[("He4", 2.0)],
        "(n,3npa)" => &[("H1", 1.0), ("He4", 1.0)],
        "(n,4n2p)" => &[("H1", 2.0)],
        "(n,4n2a)" => &[("He4", 2.0)],
        "(n,4npa)" => &[("H1", 1.0), ("He4", 1.0)],
        "(n,5n2p)" => &[("H1", 2.0)],
        "(n,3n2pa)" => &[("H1", 2.0), ("He4", 1.0)],

        _ => &[],
    }
}

/// Accumulate the Bateman transmutation matrix entries into a generic sink.
///
/// Refuse a solve that needs a chain subsection the chain was built without.
///
/// A subsection left out is invisible in the parsed chain: no reactions on a
/// nuclide reads the same as a nuclide with none, and no fission yields for it
/// reads the same as a library that evaluated none. Both then solve to a
/// plausible-looking answer with the products quietly missing -- a decay-only
/// inventory from a full irradiation, or fission that loses every product
/// while still burning its parent on the diagonal.
///
/// So the check is on the rates, which say what the calculation is actually
/// asking for, rather than on the chain, which cannot tell the difference. A
/// material carrying a fissionable nuclide at zero flux never trips it.
///
/// Only what was left out is checked. A nuclide that fissions and has no
/// evaluated yields in a subsection that *was* loaded is a normal state of the
/// data -- most nuclides with a fission cross section have no yield
/// evaluation -- and is left alone.
fn check_parts_cover_rates(
    names: &[String],
    rates: &ReactionRates,
    parts: ChainParts,
) -> Result<(), String> {
    if parts.reactions && parts.fission_yields {
        return Ok(());
    }

    // Every nuclide that asks for what is missing.
    //
    // Counted rather than ranked. Reporting whichever the walk reached first
    // named Ac225 at a reader holding a lump of uranium, and ranking by rate
    // was no better: a rate is per atom of its parent, so it put Cf251 and
    // Am242m at the top of a U235 slug on cross section alone, with nothing
    // said about how much of either is there. Weighting by abundance would fix
    // that and needs the densities, which the matrix builder is not given. So
    // the count carries the message -- 88 nuclides is unmistakably "this
    // material fissions" -- and the names are a sorted sample, offered as
    // examples rather than as the worst offenders.
    let mut demanded: Vec<(&str, &str)> = Vec::new();
    for name in names {
        let Some(per_kind) = rates.get(name.as_str()) else {
            continue;
        };
        for (kind, &rate) in per_kind {
            if rate == 0.0 {
                continue;
            }
            if !parts.reactions || (!parts.fission_yields && kind.contains("fission")) {
                demanded.push((name.as_str(), kind.as_str()));
            }
        }
    }
    if demanded.is_empty() {
        return Ok(());
    }
    demanded.sort_unstable();
    demanded.dedup();

    let listed = demanded
        .iter()
        .take(3)
        .map(|(name, kind)| format!("{name} {kind}"))
        .collect::<Vec<_>>()
        .join(", ");
    let subject = match demanded.len() {
        1 => listed.clone(),
        n if n <= 3 => format!("{n} nuclides ({listed})"),
        n => format!("{n} nuclides ({listed}, ...)"),
    };

    Err(if parts.reactions {
        format!(
            "the fission-yields subsection is not loaded, but {subject} in this \
             material {} a non-zero fission rate, so those fission products would \
             be lost while their parents still burn. Set \
             yamc.transmutation_fission_yields to a library keyword or a chain \
             directory",
            if demanded.len() == 1 { "has" } else { "have" }
        )
    } else {
        format!(
            "the reactions subsection is not loaded, but {subject} in this material \
             {} a non-zero reaction rate. Set yamc.transmutation_reactions to a \
             library keyword or a chain directory, or drive this material with no \
             reaction rates for a decay-only calculation",
            if demanded.len() == 1 { "has" } else { "have" }
        )
    })
}

/// Encodes the shared logic for both `build_matrix()` (dense) and
/// `build_matrix_triplets()` (sparse COO): for each source nuclide `col`,
/// it emits production entries `(row, col, value)` for decays, grouped
/// reactions, light-particle products, and fission yields, plus the
/// diagonal loss term as `sink(col, col, -loss)`.
///
/// The `sink` closure decides how each entry is stored, so the two public
/// builders are thin wrappers over this routine.
///
/// # Errors
/// A nuclide that has fission yields and a non-zero fission rate but no entry
/// in `fy_weights`. Falling back to a single tabulated energy there would
/// silently reintroduce the spectrum-independent yields of issue #379, so it is
/// reported as the caller error it is.
///
/// # Returns
/// The matrix dimension `n` (equal to `names.len()`).
fn accumulate_matrix<F>(
    chain: &HashMap<String, ChainNuclide>,
    names: &[String],
    rates: &ReactionRates,
    fy_weights: &FissionYieldWeights,
    parts: ChainParts,
    mut sink: F,
) -> Result<usize, String>
where
    F: FnMut(usize, usize, f64),
{
    check_parts_cover_rates(names, rates, parts)?;

    let n = names.len();
    let mut index: HashMap<&str, usize> = HashMap::with_capacity(n);
    for (i, name) in names.iter().enumerate() {
        index.insert(name.as_str(), i);
    }

    // Scratch for folding a nuclide's tabulated yield vectors into one
    // effective vector before emitting. Indexed by matrix row and reused across
    // nuclides, so the matrix keeps one entry per fission product however many
    // tabulated energies contribute to it.
    let mut fy_fold = vec![0.0f64; n];
    let mut fy_touched: Vec<usize> = Vec::new();

    let ln2 = std::f64::consts::LN_2;

    for (col, name) in names.iter().enumerate() {
        let nuc = match chain.get(name) {
            Some(n) => n,
            None => continue,
        };
        let mut loss = 0.0;

        // Decay contributions
        if let Some(half_life) = nuc.half_life {
            if half_life > 0.0 {
                let lambda = ln2 / half_life;
                loss += lambda;
                for decay in &nuc.decays {
                    if let Some(target) = &decay.target {
                        if let Some(&row) = index.get(target.as_str()) {
                            sink(row, col, decay.branching * lambda);
                        }
                    }
                    // Add light particle production from decay (He4 from alpha decay, etc.)
                    for &(particle, count) in decay_particle_products(&decay.kind) {
                        if let Some(&row) = index.get(particle) {
                            sink(row, col, decay.branching * lambda * count);
                        }
                    }
                }
            }
        }

        // Reaction contributions.
        //
        // A `BTreeMap` rather than a `HashMap`, because this is iterated and
        // two float accumulations downstream take their order from it: `loss`
        // below sums one `rate` per kind, and every `sink` call the loop makes
        // lands in a matrix cell its caller accumulates into. Rust seeds each
        // `HashMap` instance separately, so a fresh one here walked its kinds
        // in a different order on each call and `Material.transmute()` returned
        // two different inventories for identical inputs -- irregularly, within
        // a single process, with no Monte Carlo anywhere in the path (issue
        // #502). Sorting by kind costs nothing at these sizes and makes the
        // whole path bit-reproducible.
        let mut grouped: BTreeMap<&str, Vec<&ChainReaction>> = BTreeMap::new();
        for rx in &nuc.reactions {
            grouped.entry(rx.kind.as_str()).or_default().push(rx);
        }
        if let Some(rate_map) = rates.get(name.as_str()) {
            for (rx_type, rx_list) in &grouped {
                let rate = *rate_map.get(*rx_type).unwrap_or(&0.0);
                if rate == 0.0 {
                    continue;
                }
                loss += rate;
                for rx in rx_list {
                    if let Some(target) = &rx.target {
                        if let Some(&row) = index.get(target.as_str()) {
                            sink(row, col, rx.branching * rate);
                        }
                    }
                    // Add light particle production (H1, He4, etc.)
                    for &(particle, count) in light_particle_products(&rx.kind) {
                        if let Some(&row) = index.get(particle) {
                            sink(row, col, rx.branching * rate * count);
                        }
                    }
                    // Add fission product contributions, folding the tabulated
                    // yield vectors with this material's spectrum weights
                    // (issue #379). Products are accumulated into `fy_fold`
                    // first so each one reaches the sink once, leaving the
                    // sparsity pattern the same as a single-energy yield.
                    if rx.kind.contains("fission") {
                        if let Some(fy_set) = &nuc.fission_yields {
                            let weights = fy_weights.get(name.as_str()).ok_or_else(|| {
                                format!(
                                    "no fission-yield spectrum weights for {name}, which has \
                                     yields and a non-zero {} rate; the driver must compute \
                                     them from the flux",
                                    rx.kind
                                )
                            })?;
                            for (k, &c_k) in weights.iter().enumerate() {
                                if c_k == 0.0 {
                                    continue;
                                }
                                let Some(fy) = fy_set.yields.get(k) else {
                                    return Err(format!(
                                        "{name} has {} fission-yield spectrum weights but {} \
                                         tabulated energies",
                                        weights.len(),
                                        fy_set.yields.len()
                                    ));
                                };
                                for (product, yield_val) in &fy.products {
                                    if let Some(&row) = index.get(product.as_str()) {
                                        if fy_fold[row] == 0.0 {
                                            fy_touched.push(row);
                                        }
                                        fy_fold[row] += c_k * yield_val;
                                    }
                                }
                            }
                            for &row in &fy_touched {
                                sink(row, col, rx.branching * rate * fy_fold[row]);
                                fy_fold[row] = 0.0;
                            }
                            fy_touched.clear();
                        }
                    }
                }
            }
        }

        if loss != 0.0 {
            // Diagonal loss term: `-= loss` is equivalent to `+= -loss`.
            sink(col, col, -loss);
        }
    }

    Ok(n)
}

/// Build the transmutation matrix from chain data and reaction rates.
///
/// The matrix A encodes the Bateman equations:
///   dN_i/dt = sum_j (lambda_j->i + sigma_j->i*phi) * N_j - (lambda_i + sigma_i*phi) * N_i
///
/// Matrix layout: A[row * n + col] where row is the production target
/// and col is the source nuclide. Diagonal entries are negative (loss terms).
///
/// # Arguments
/// * `chain` - Parsed transmutation chain
/// * `names` - Ordered list of nuclide names (defines matrix indices)
/// * `rates` - Reaction rates: nuclide -> reaction_type -> rate [1/s]
/// * `fy_weights` - Per-nuclide spectrum weights over the tabulated fission-yield
///   energies (see [`FissionYieldWeights`])
///
/// # Returns
/// Flattened n x n matrix (row-major)
pub fn build_matrix(
    chain: &HashMap<String, ChainNuclide>,
    names: &[String],
    rates: &ReactionRates,
    fy_weights: &FissionYieldWeights,
    parts: ChainParts,
) -> Result<Vec<f64>, String> {
    let n = names.len();
    let mut a = vec![0.0f64; n * n];
    accumulate_matrix(chain, names, rates, fy_weights, parts, |row, col, value| {
        a[row * n + col] += value;
    })?;
    Ok(a)
}

/// Build the transmutation matrix as COO triplets for sparse solvers.
///
/// Same Bateman equation logic as `build_matrix()`, but returns
/// (row, col, value) triplets instead of a dense matrix.
///
/// Note: may contain duplicate (row, col) entries -- the sparse matrix
/// constructor should sum them.
///
/// # Returns
/// `(triplets, n)` where `triplets` is `Vec<(usize, usize, f64)>` and `n` is the dimension.
pub fn build_matrix_triplets(
    chain: &HashMap<String, ChainNuclide>,
    names: &[String],
    rates: &ReactionRates,
    fy_weights: &FissionYieldWeights,
    parts: ChainParts,
) -> Result<MatrixTriplets, String> {
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    let n = accumulate_matrix(chain, names, rates, fy_weights, parts, |row, col, value| {
        triplets.push((row, col, value));
    })?;
    Ok((triplets, n))
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod parts_tests {
    //! A subsection left out must be refused when the solve needs it, and
    //! ignored when it does not.

    use super::*;

    fn rates(pairs: &[(&str, &str, f64)]) -> ReactionRates {
        let mut out: ReactionRates = HashMap::new();
        for &(nuclide, kind, rate) in pairs {
            out.entry(nuclide.to_string())
                .or_default()
                .insert(kind.to_string(), rate);
        }
        out
    }

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    const OFF: ChainParts = ChainParts {
        reactions: false,
        fission_yields: false,
    };

    #[test]
    fn a_complete_chain_is_never_questioned() {
        let names = names(&["U235"]);
        let rates = rates(&[("U235", "fission", 1e-5)]);
        assert!(check_parts_cover_rates(&names, &rates, ChainParts::default()).is_ok());
    }

    #[test]
    fn no_reactions_and_no_rates_is_a_decay_only_run() {
        // The case this is all for: cooling an inventory with the reaction
        // subsections left out. Nothing asks for them, so nothing complains.
        let names = names(&["Co60"]);
        assert!(check_parts_cover_rates(&names, &HashMap::new(), OFF).is_ok());
    }

    #[test]
    fn a_zero_rate_does_not_demand_a_subsection() {
        // A fissionable nuclide sitting in a material at no flux is not a
        // calculation that needs fission yields.
        let names = names(&["U235"]);
        let rates = rates(&[("U235", "fission", 0.0)]);
        assert!(check_parts_cover_rates(&names, &rates, OFF).is_ok());
    }

    #[test]
    fn a_rate_with_no_reactions_loaded_is_refused() {
        let names = names(&["Fe56"]);
        let rates = rates(&[("Fe56", "(n,p)", 1e-9)]);
        let parts = ChainParts {
            reactions: false,
            fission_yields: true,
        };
        let err = check_parts_cover_rates(&names, &rates, parts).expect_err("refused");
        assert!(err.contains("Fe56"), "{err}");
        assert!(err.contains("(n,p)"), "{err}");
        assert!(err.contains("transmutation_reactions"), "{err}");
    }

    #[test]
    fn a_fission_rate_with_no_yields_loaded_is_refused() {
        let names = names(&["U235"]);
        let rates = rates(&[("U235", "fission", 1e-5)]);
        let parts = ChainParts {
            reactions: true,
            fission_yields: false,
        };
        let err = check_parts_cover_rates(&names, &rates, parts).expect_err("refused");
        assert!(err.contains("U235"), "{err}");
        assert!(err.contains("transmutation_fission_yields"), "{err}");
    }

    #[test]
    fn a_non_fission_rate_does_not_need_yields() {
        // Fission yields off is the ordinary setting for a fusion material.
        // Its (n,2n) and (n,p) rates must not be caught by the fission guard.
        let names = names(&["W186"]);
        let rates = rates(&[("W186", "(n,2n)", 3e-9), ("W186", "(n,gamma)", 1e-10)]);
        let parts = ChainParts {
            reactions: true,
            fission_yields: false,
        };
        assert!(check_parts_cover_rates(&names, &rates, parts).is_ok());
    }

    /// The count is the message. Naming a few is a sample, in a fixed order so
    /// the same run says the same thing twice, and explicitly not a ranking:
    /// a rate is per atom of its parent, so ordering by it puts a trace minor
    /// actinide above the U235 the reader actually loaded.
    #[test]
    fn the_offenders_are_counted_and_a_sample_named() {
        let names = names(&["Ac225", "Pu239", "U235", "Cm244"]);
        let rates = rates(&[
            ("Ac225", "fission", 1e-12),
            ("Pu239", "fission", 4e-6),
            ("U235", "fission", 9e-6),
            ("Cm244", "fission", 2e-9),
        ]);
        let parts = ChainParts {
            reactions: true,
            fission_yields: false,
        };
        let err = check_parts_cover_rates(&names, &rates, parts).expect_err("refused");
        assert!(err.contains("4 nuclides"), "{err}");
        assert!(
            err.contains("(Ac225 fission, Cm244 fission, Pu239 fission, ...)"),
            "{err}"
        );
        assert!(err.contains("have a non-zero fission rate"), "{err}");
    }

    /// Whatever the rates are, the same set says the same thing.
    #[test]
    fn the_message_does_not_depend_on_the_rates() {
        let names = names(&["Ac225", "Pu239", "U235", "Cm244"]);
        let quiet = rates(&[
            ("Ac225", "fission", 9e-6),
            ("Pu239", "fission", 1e-12),
            ("U235", "fission", 2e-9),
            ("Cm244", "fission", 4e-6),
        ]);
        let loud = rates(&[
            ("Ac225", "fission", 1e-12),
            ("Pu239", "fission", 4e-6),
            ("U235", "fission", 9e-6),
            ("Cm244", "fission", 2e-9),
        ]);
        let parts = ChainParts {
            reactions: true,
            fission_yields: false,
        };
        assert_eq!(
            check_parts_cover_rates(&names, &quiet, parts),
            check_parts_cover_rates(&names, &loud, parts)
        );
    }

    /// One offender reads as one, not as "1 nuclides (U235 fission) have".
    #[test]
    fn a_single_offender_is_named_alone() {
        let names = names(&["U235"]);
        let rates = rates(&[("U235", "fission", 9e-6)]);
        let parts = ChainParts {
            reactions: true,
            fission_yields: false,
        };
        let err = check_parts_cover_rates(&names, &rates, parts).expect_err("refused");
        assert!(
            err.contains("U235 fission in this material has a non-zero fission rate"),
            "{err}"
        );
        assert!(!err.contains("nuclides"), "{err}");
    }

    #[test]
    fn a_rate_on_a_nuclide_outside_the_matrix_is_not_read() {
        // `names` is the matrix's own nuclide set. A rate for something not in
        // it contributes nothing, so it cannot make the solve wrong either.
        let names = names(&["Fe56"]);
        let rates = rates(&[("U235", "fission", 1e-5)]);
        assert!(check_parts_cover_rates(&names, &rates, OFF).is_ok());
    }

    #[test]
    fn the_builders_refuse_before_they_allocate() {
        // Through the public entry point, so the guard cannot be bypassed by
        // the path the steppers actually take.
        let chain: HashMap<String, ChainNuclide> = HashMap::new();
        let names = names(&["U235"]);
        let rates = rates(&[("U235", "fission", 1e-5)]);
        let parts = ChainParts {
            reactions: true,
            fission_yields: false,
        };
        assert!(build_matrix_triplets(&chain, &names, &rates, &HashMap::new(), parts).is_err());
        assert!(build_matrix(&chain, &names, &rates, &HashMap::new(), parts).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{ChainNuclide, ChainReaction, FissionYield, FissionYieldSet};
    use std::sync::Arc;

    /// No fissionable nuclide in the fixture, so the builder demands nothing.
    fn fy_weights() -> FissionYieldWeights {
        HashMap::new()
    }

    /// All the weight on each named nuclide's single tabulated energy, which
    /// is what a fixture with one yield point means.
    fn single_energy_weights(nuclides: &[&str]) -> FissionYieldWeights {
        nuclides
            .iter()
            .map(|n| ((*n).to_string(), vec![1.0]))
            .collect()
    }

    /// A nuclide with only the fields a matrix test cares about.
    ///
    /// These tests are about the shape of the burnup matrix, not about decay
    /// data, so they set a name, a half-life and a decay list and want nothing
    /// to do with the rest. Spelling every field out 39 times meant each new
    /// field on `ChainNuclide` was 39 mechanical edits that read as though the
    /// tests had an opinion about it. They do not.
    fn nuclide(
        name: &str,
        half_life: Option<f64>,
        decays: Vec<ChainReaction>,
        reactions: Vec<ChainReaction>,
    ) -> ChainNuclide {
        ChainNuclide {
            name: name.to_string(),
            half_life,
            half_life_uncertainty: None,
            decay_energy: 0.0,
            reactions,
            decays,
            fission_yields: None,
            sources: Vec::new(),
            decay_energy_uncertainty: None,
        }
    }

    #[test]
    fn test_build_matrix_decay_only() {
        // Simple A -> B decay
        let mut chain = HashMap::new();
        let half_life = 3600.0; // 1 hour
        chain.insert(
            "A".to_string(),
            nuclide(
                "A",
                Some(half_life),
                vec![ChainReaction {
                    kind: "beta-".to_string(),
                    target: Some("B".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                vec![],
            ),
        );
        chain.insert("B".to_string(), nuclide("B", None, vec![], vec![]));

        let names = vec!["A".to_string(), "B".to_string()];
        let rates: ReactionRates = HashMap::new();
        let matrix =
            build_matrix(&chain, &names, &rates, &fy_weights(), ChainParts::default()).unwrap();

        let lambda = std::f64::consts::LN_2 / half_life;
        // A[0,0] = -lambda (loss of A)
        assert!((matrix[0] - (-lambda)).abs() < 1e-15);
        // A[1,0] = lambda (production of B from A)
        assert!((matrix[2] - lambda).abs() < 1e-15);
        // A[0,1] = 0, A[1,1] = 0 (B is stable)
        assert_eq!(matrix[1], 0.0);
        assert_eq!(matrix[3], 0.0);
    }

    #[test]
    fn test_build_matrix_with_reaction() {
        // A + n -> B (n,gamma)
        let mut chain = HashMap::new();
        chain.insert(
            "A".to_string(),
            nuclide(
                "A",
                None,
                vec![],
                vec![ChainReaction {
                    kind: "(n,gamma)".to_string(),
                    target: Some("B".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
            ),
        );
        chain.insert("B".to_string(), nuclide("B", None, vec![], vec![]));

        let names = vec!["A".to_string(), "B".to_string()];
        let mut rates: ReactionRates = HashMap::new();
        let rate = 1.0e-5; // sigma * phi
        let mut a_rates = HashMap::new();
        a_rates.insert("(n,gamma)".to_string(), rate);
        rates.insert("A".to_string(), a_rates);

        let matrix =
            build_matrix(&chain, &names, &rates, &fy_weights(), ChainParts::default()).unwrap();

        // A[0,0] = -rate (loss of A)
        assert!((matrix[0] - (-rate)).abs() < 1e-20);
        // A[1,0] = rate (production of B from A)
        assert!((matrix[2] - rate).abs() < 1e-20);
    }

    /// The per-edge rates are the matrix's own products, kept rather than summed.
    #[test]
    fn test_per_edge_rates_splits_a_rate_over_its_targets() {
        let mut chain = HashMap::new();
        chain.insert(
            "A".to_string(),
            nuclide(
                "A",
                None,
                vec![],
                vec![
                    // One kind, two final states: an isomeric split.
                    ChainReaction {
                        kind: "(n,gamma)".to_string(),
                        target: Some("B".to_string()),
                        branching: 0.75,
                        q_value: None,
                    },
                    ChainReaction {
                        kind: "(n,gamma)".to_string(),
                        target: Some("B_m1".to_string()),
                        branching: 0.25,
                        q_value: None,
                    },
                    ChainReaction {
                        kind: "(n,2n)".to_string(),
                        target: Some("C".to_string()),
                        branching: 1.0,
                        q_value: None,
                    },
                    // No named product, and no rate given for it below.
                    ChainReaction {
                        kind: "fission".to_string(),
                        target: None,
                        branching: 1.0,
                        q_value: None,
                    },
                ],
            ),
        );

        let mut rates: ReactionRates = HashMap::new();
        rates.insert(
            "A".to_string(),
            HashMap::from([
                ("(n,gamma)".to_string(), 4.0e-9),
                ("(n,2n)".to_string(), 1.0e-11),
            ]),
        );

        let edges = per_edge_rates(&chain, &rates);

        let gamma = &edges["A"]["(n,gamma)"];
        assert_eq!(gamma.len(), 2);
        let b = gamma
            .iter()
            .find(|(t, _)| t.as_deref() == Some("B"))
            .unwrap()
            .1;
        let bm = gamma
            .iter()
            .find(|(t, _)| t.as_deref() == Some("B_m1"))
            .unwrap()
            .1;
        assert!((b - 0.75 * 4.0e-9).abs() < 1e-24);
        assert!((bm - 0.25 * 4.0e-9).abs() < 1e-24);
        // The split partitions the channel: the edges sum back to its rate.
        assert!((b + bm - 4.0e-9).abs() < 1e-24);

        assert_eq!(edges["A"]["(n,2n)"], vec![(Some("C".to_string()), 1.0e-11)]);
        // A channel the flux never drove is absent, not zero.
        assert!(!edges["A"].contains_key("fission"));
    }

    /// Fission keeps its rate under a `None` target: the channel is where the
    /// rate went, even though the products come from the yields.
    #[test]
    fn test_per_edge_rates_reports_fission_without_a_target() {
        let mut chain = HashMap::new();
        chain.insert(
            "A".to_string(),
            nuclide(
                "A",
                None,
                vec![],
                vec![ChainReaction {
                    kind: "fission".to_string(),
                    target: None,
                    branching: 1.0,
                    q_value: None,
                }],
            ),
        );
        let rates: ReactionRates = HashMap::from([(
            "A".to_string(),
            HashMap::from([("fission".to_string(), 2.5e-10)]),
        )]);

        assert_eq!(
            per_edge_rates(&chain, &rates)["A"]["fission"],
            vec![(None, 2.5e-10)]
        );
    }

    /// A decay-only step passes no rates, so there are no edges to report.
    #[test]
    fn test_per_edge_rates_is_empty_without_rates() {
        let chain = simple_two_nuclide_chain();
        assert!(per_edge_rates(&chain, &HashMap::new()).is_empty());
    }

    /// A rate for a nuclide the chain does not carry is ignored rather than
    /// inventing a parent.
    #[test]
    fn test_per_edge_rates_ignores_a_parent_outside_the_chain() {
        let chain = simple_two_nuclide_chain();
        let rates: ReactionRates = HashMap::from([(
            "Zz999".to_string(),
            HashMap::from([("(n,gamma)".to_string(), 1.0)]),
        )]);
        assert!(per_edge_rates(&chain, &rates).is_empty());
    }

    /// A -> B on `(n,gamma)`, nothing else.
    fn simple_two_nuclide_chain() -> HashMap<String, ChainNuclide> {
        HashMap::from([
            (
                "A".to_string(),
                nuclide(
                    "A",
                    None,
                    vec![],
                    vec![ChainReaction {
                        kind: "(n,gamma)".to_string(),
                        target: Some("B".to_string()),
                        branching: 1.0,
                        q_value: None,
                    }],
                ),
            ),
            ("B".to_string(), nuclide("B", None, vec![], vec![])),
        ])
    }

    #[test]
    fn test_fission_yield_in_matrix() {
        // A undergoes fission producing B (yield=1.0) and C (yield=1.0)
        let mut chain = HashMap::new();
        chain.insert(
            "A".to_string(),
            ChainNuclide {
                name: "A".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: vec![ChainReaction {
                    kind: "fission".to_string(),
                    target: None,
                    branching: 1.0,
                    q_value: None,
                }],
                decays: vec![],
                fission_yields: Some(Arc::new(FissionYieldSet {
                    yields: vec![FissionYield {
                        energy: 0.0253,
                        products: vec![("B".to_string(), 1.0), ("C".to_string(), 1.0)],
                    }],
                })),
                sources: Vec::new(),
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        chain.insert("B".to_string(), nuclide("B", None, vec![], vec![]));
        chain.insert("C".to_string(), nuclide("C", None, vec![], vec![]));

        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut rates: ReactionRates = HashMap::new();
        let mut a_rates = HashMap::new();
        a_rates.insert("fission".to_string(), 1e-5);
        rates.insert("A".to_string(), a_rates);

        let matrix = build_matrix(
            &chain,
            &names,
            &rates,
            &single_energy_weights(&["A"]),
            ChainParts::default(),
        )
        .unwrap();
        // A[0,0] = -1e-5 (loss from fission)
        assert!((matrix[0] - (-1e-5)).abs() < 1e-20);
        // A[1,0] = 1e-5 * 1.0 (B produced from A fission, yield=1.0)
        assert!((matrix[3] - 1e-5).abs() < 1e-20);
        // A[2,0] = 1e-5 * 1.0 (C produced from A fission, yield=1.0)
        assert!((matrix[6] - 1e-5).abs() < 1e-20);
    }

    #[test]
    fn test_fission_yield_with_fractional_branching() {
        // Fission with branching_ratio < 1.0 (e.g. two fission channels)
        let mut chain = HashMap::new();
        chain.insert(
            "U".to_string(),
            ChainNuclide {
                name: "U".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: vec![ChainReaction {
                    kind: "(n,fission)".to_string(),
                    target: None,
                    branching: 0.5,
                    q_value: None,
                }],
                decays: vec![],
                fission_yields: Some(Arc::new(FissionYieldSet {
                    yields: vec![FissionYield {
                        energy: 0.0253,
                        products: vec![("Xe".to_string(), 0.06), ("Sr".to_string(), 0.04)],
                    }],
                })),
                sources: Vec::new(),
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        chain.insert("Xe".to_string(), nuclide("Xe", None, vec![], vec![]));
        chain.insert("Sr".to_string(), nuclide("Sr", None, vec![], vec![]));

        let names = vec!["U".to_string(), "Xe".to_string(), "Sr".to_string()];
        let mut rates: ReactionRates = HashMap::new();
        let mut u_rates = HashMap::new();
        let rate = 2e-5;
        u_rates.insert("(n,fission)".to_string(), rate);
        rates.insert("U".to_string(), u_rates);

        let matrix = build_matrix(
            &chain,
            &names,
            &rates,
            &single_energy_weights(&["U"]),
            ChainParts::default(),
        )
        .unwrap();
        // loss = rate (whole rate, not branching-scaled)
        assert!((matrix[0] - (-rate)).abs() < 1e-20);
        // Xe production = branching * rate * yield = 0.5 * 2e-5 * 0.06
        let xe_expected = 0.5 * rate * 0.06;
        assert!(
            (matrix[3] - xe_expected).abs() < 1e-25,
            "Xe production: expected {xe_expected}, got {}",
            matrix[3]
        );
        // Sr production = 0.5 * 2e-5 * 0.04
        let sr_expected = 0.5 * rate * 0.04;
        assert!(
            (matrix[6] - sr_expected).abs() < 1e-25,
            "Sr production: expected {sr_expected}, got {}",
            matrix[6]
        );
    }

    #[test]
    fn test_alpha_decay_produces_he4() {
        // U -> Th via alpha decay, should also produce He4
        let half_life = 1e10;
        let mut chain = HashMap::new();
        chain.insert(
            "U".to_string(),
            nuclide(
                "U",
                Some(half_life),
                vec![ChainReaction {
                    kind: "alpha".to_string(),
                    target: Some("Th".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                vec![],
            ),
        );
        chain.insert("Th".to_string(), nuclide("Th", None, vec![], vec![]));
        chain.insert("He4".to_string(), nuclide("He4", None, vec![], vec![]));

        let names = vec!["U".to_string(), "Th".to_string(), "He4".to_string()];
        let rates: ReactionRates = HashMap::new();
        let matrix =
            build_matrix(&chain, &names, &rates, &fy_weights(), ChainParts::default()).unwrap();

        let lambda = std::f64::consts::LN_2 / half_life;
        // Diagonal: loss of U
        assert!((matrix[0] - (-lambda)).abs() < 1e-30);
        // Off-diagonal: Th production from U
        assert!((matrix[3] - lambda).abs() < 1e-30);
        // Off-diagonal: He4 production from U alpha decay
        assert!(
            (matrix[6] - lambda).abs() < 1e-30,
            "He4 should be produced from alpha decay, expected {lambda}, got {}",
            matrix[6]
        );
    }

    #[test]
    fn test_np_reaction_produces_h1() {
        // A + n -> B + p: the (n,p) reaction should produce H1
        let mut chain = HashMap::new();
        chain.insert(
            "A".to_string(),
            nuclide(
                "A",
                None,
                vec![],
                vec![ChainReaction {
                    kind: "(n,p)".to_string(),
                    target: Some("B".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
            ),
        );
        chain.insert("B".to_string(), nuclide("B", None, vec![], vec![]));
        chain.insert("H1".to_string(), nuclide("H1", None, vec![], vec![]));

        let names = vec!["A".to_string(), "B".to_string(), "H1".to_string()];
        let rate = 3e-6;
        let mut rates: ReactionRates = HashMap::new();
        let mut a_rates = HashMap::new();
        a_rates.insert("(n,p)".to_string(), rate);
        rates.insert("A".to_string(), a_rates);

        let matrix =
            build_matrix(&chain, &names, &rates, &fy_weights(), ChainParts::default()).unwrap();
        // Loss of A
        assert!((matrix[0] - (-rate)).abs() < 1e-25);
        // B production
        assert!((matrix[3] - rate).abs() < 1e-25);
        // H1 production from (n,p)
        assert!(
            (matrix[6] - rate).abs() < 1e-25,
            "H1 should be produced from (n,p), expected {rate}, got {}",
            matrix[6]
        );
    }

    #[test]
    fn test_na_reaction_produces_he4() {
        // A + n -> B + alpha: the (n,a) reaction should produce He4
        let mut chain = HashMap::new();
        chain.insert(
            "A".to_string(),
            nuclide(
                "A",
                None,
                vec![],
                vec![ChainReaction {
                    kind: "(n,a)".to_string(),
                    target: Some("B".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
            ),
        );
        chain.insert("B".to_string(), nuclide("B", None, vec![], vec![]));
        chain.insert("He4".to_string(), nuclide("He4", None, vec![], vec![]));

        let names = vec!["A".to_string(), "B".to_string(), "He4".to_string()];
        let rate = 5e-7;
        let mut rates: ReactionRates = HashMap::new();
        let mut a_rates = HashMap::new();
        a_rates.insert("(n,a)".to_string(), rate);
        rates.insert("A".to_string(), a_rates);

        let matrix =
            build_matrix(&chain, &names, &rates, &fy_weights(), ChainParts::default()).unwrap();
        assert!((matrix[6] - rate).abs() < 1e-25, "He4 from (n,a)");
    }

    #[test]
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn test_combined_decay_and_reaction() {
        // `matrix[row * n + col]` is intentional row-major indexing for
        // readability; the row=0 / col=0 cases trip erasing_op / identity_op.
        // Ag108 scenario: decays (beta- to Cd108) + reactions ((n,p) to Pd108)
        let half_life = 142.92;
        let mut chain = HashMap::new();
        chain.insert(
            "Ag108".to_string(),
            nuclide(
                "Ag108",
                Some(half_life),
                vec![ChainReaction {
                    kind: "beta-".to_string(),
                    target: Some("Cd108".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                vec![ChainReaction {
                    kind: "(n,p)".to_string(),
                    target: Some("Pd108".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
            ),
        );
        chain.insert(
            "Cd108".to_string(),
            nuclide(
                "Cd108",
                None,
                vec![],
                vec![ChainReaction {
                    kind: "(n,a)".to_string(),
                    target: Some("Pd105".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
            ),
        );
        chain.insert("Pd108".to_string(), nuclide("Pd108", None, vec![], vec![]));
        chain.insert("Pd105".to_string(), nuclide("Pd105", None, vec![], vec![]));
        chain.insert("H1".to_string(), nuclide("H1", None, vec![], vec![]));
        chain.insert("He4".to_string(), nuclide("He4", None, vec![], vec![]));

        let names = vec![
            "Ag108".to_string(),
            "Cd108".to_string(),
            "Pd108".to_string(),
            "Pd105".to_string(),
            "H1".to_string(),
            "He4".to_string(),
        ];
        let n = names.len();

        let ag_np_rate = 1e-6;
        let cd_na_rate = 2e-7;
        let mut rates: ReactionRates = HashMap::new();
        let mut ag_rates = HashMap::new();
        ag_rates.insert("(n,p)".to_string(), ag_np_rate);
        rates.insert("Ag108".to_string(), ag_rates);
        let mut cd_rates = HashMap::new();
        cd_rates.insert("(n,a)".to_string(), cd_na_rate);
        rates.insert("Cd108".to_string(), cd_rates);

        let matrix =
            build_matrix(&chain, &names, &rates, &fy_weights(), ChainParts::default()).unwrap();
        let lambda = std::f64::consts::LN_2 / half_life;

        // Ag108 loss = decay + (n,p) rate
        let ag108_loss = lambda + ag_np_rate;
        assert!(
            (matrix[0 * n + 0] - (-ag108_loss)).abs() < 1e-15,
            "Ag108 diagonal loss"
        );

        // Cd108 production from Ag108 beta- decay
        assert!(
            (matrix[1 * n + 0] - lambda).abs() < 1e-15,
            "Cd108 from Ag108 beta- decay"
        );

        // Pd108 production from Ag108 (n,p)
        assert!(
            (matrix[2 * n + 0] - ag_np_rate).abs() < 1e-25,
            "Pd108 from Ag108 (n,p)"
        );

        // H1 production from Ag108 (n,p) light particle
        assert!(
            (matrix[4 * n + 0] - ag_np_rate).abs() < 1e-25,
            "H1 from Ag108 (n,p)"
        );

        // Cd108 loss = (n,a) rate (no decay)
        assert!(
            (matrix[1 * n + 1] - (-cd_na_rate)).abs() < 1e-25,
            "Cd108 diagonal loss"
        );

        // Pd105 from Cd108 (n,a)
        assert!(
            (matrix[3 * n + 1] - cd_na_rate).abs() < 1e-25,
            "Pd105 from Cd108 (n,a)"
        );

        // He4 from Cd108 (n,a)
        assert!(
            (matrix[5 * n + 1] - cd_na_rate).abs() < 1e-25,
            "He4 from Cd108 (n,a)"
        );
    }

    #[test]
    fn test_triplets_match_dense_for_fission() {
        // Verify build_matrix_triplets produces the same result as build_matrix
        // for a system with fission yields
        let mut chain = HashMap::new();
        chain.insert(
            "U".to_string(),
            ChainNuclide {
                name: "U".to_string(),
                half_life: Some(1e16),
                decay_energy: 0.0,
                reactions: vec![
                    ChainReaction {
                        kind: "(n,gamma)".to_string(),
                        target: Some("U2".to_string()),
                        branching: 1.0,
                        q_value: None,
                    },
                    ChainReaction {
                        kind: "(n,fission)".to_string(),
                        target: None,
                        branching: 1.0,
                        q_value: None,
                    },
                ],
                decays: vec![ChainReaction {
                    kind: "alpha".to_string(),
                    target: Some("Th".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                fission_yields: Some(Arc::new(FissionYieldSet {
                    yields: vec![FissionYield {
                        energy: 0.0253,
                        products: vec![("Xe".to_string(), 0.065), ("Cs".to_string(), 0.062)],
                    }],
                })),
                sources: Vec::new(),
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        for name in &["U2", "Th", "He4", "Xe", "Cs"] {
            chain.insert(name.to_string(), nuclide(name, None, vec![], vec![]));
        }

        let names: Vec<String> = vec!["U", "U2", "Th", "He4", "Xe", "Cs"]
            .into_iter()
            .map(String::from)
            .collect();
        let n = names.len();

        let mut rates: ReactionRates = HashMap::new();
        let mut u_rates = HashMap::new();
        u_rates.insert("(n,gamma)".to_string(), 1e-6);
        u_rates.insert("(n,fission)".to_string(), 2e-5);
        rates.insert("U".to_string(), u_rates);

        let fy = single_energy_weights(&["U"]);
        let dense = build_matrix(&chain, &names, &rates, &fy, ChainParts::default()).unwrap();
        let (triplets, tn) =
            build_matrix_triplets(&chain, &names, &rates, &fy, ChainParts::default()).unwrap();
        assert_eq!(tn, n);

        // Reconstruct dense from triplets
        let mut from_triplets = vec![0.0f64; n * n];
        for (row, col, val) in &triplets {
            from_triplets[row * n + col] += val;
        }

        for i in 0..n * n {
            let row = i / n;
            let col = i % n;
            assert!(
                (dense[i] - from_triplets[i]).abs() < 1e-25,
                "Mismatch at [{row},{col}]: dense={}, triplets={}",
                dense[i],
                from_triplets[i]
            );
        }
    }

    #[test]
    fn test_nuclide_not_in_chain_skipped() {
        // If a nuclide name is in names but not in chain, it should be skipped
        let chain: HashMap<String, ChainNuclide> = HashMap::new();
        let names = vec!["Missing".to_string()];
        let rates: ReactionRates = HashMap::new();

        let matrix =
            build_matrix(&chain, &names, &rates, &fy_weights(), ChainParts::default()).unwrap();
        assert_eq!(matrix.len(), 1);
        assert_eq!(matrix[0], 0.0, "Missing nuclide should contribute nothing");
    }

    #[test]
    fn test_no_rates_decay_only() {
        // With empty rates, only decay should contribute
        let half_life = 3600.0;
        let mut chain = HashMap::new();
        chain.insert(
            "A".to_string(),
            nuclide(
                "A",
                Some(half_life),
                vec![ChainReaction {
                    kind: "beta-".to_string(),
                    target: Some("B".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                vec![ChainReaction {
                    kind: "(n,gamma)".to_string(),
                    target: Some("B".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
            ),
        );
        chain.insert("B".to_string(), nuclide("B", None, vec![], vec![]));

        let names = vec!["A".to_string(), "B".to_string()];
        let rates: ReactionRates = HashMap::new(); // no rates

        let matrix =
            build_matrix(&chain, &names, &rates, &fy_weights(), ChainParts::default()).unwrap();
        let lambda = std::f64::consts::LN_2 / half_life;

        // Only decay, no reaction contribution
        assert!(
            (matrix[0] - (-lambda)).abs() < 1e-20,
            "Loss should be decay only"
        );
        assert!(
            (matrix[2] - lambda).abs() < 1e-20,
            "Production should be decay only"
        );
    }

    #[test]
    fn test_fission_product_outside_names_ignored() {
        // If fission produces a nuclide not in names[], it should be silently ignored
        let mut chain = HashMap::new();
        chain.insert(
            "U".to_string(),
            ChainNuclide {
                name: "U".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: vec![ChainReaction {
                    kind: "fission".to_string(),
                    target: None,
                    branching: 1.0,
                    q_value: None,
                }],
                decays: vec![],
                fission_yields: Some(Arc::new(FissionYieldSet {
                    yields: vec![FissionYield {
                        energy: 0.0253,
                        products: vec![("Xe".to_string(), 0.065), ("NotInNames".to_string(), 0.05)],
                    }],
                })),
                sources: Vec::new(),
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        chain.insert("Xe".to_string(), nuclide("Xe", None, vec![], vec![]));

        // NotInNames is NOT in the names list
        let names = vec!["U".to_string(), "Xe".to_string()];
        let rate = 1e-5;
        let mut rates: ReactionRates = HashMap::new();
        let mut u_rates = HashMap::new();
        u_rates.insert("fission".to_string(), rate);
        rates.insert("U".to_string(), u_rates);

        let matrix = build_matrix(
            &chain,
            &names,
            &rates,
            &single_energy_weights(&["U"]),
            ChainParts::default(),
        )
        .unwrap();
        // Should not panic; Xe should still get its contribution
        assert!((matrix[0] - (-rate)).abs() < 1e-20, "U loss");
        assert!(
            (matrix[2] - rate * 0.065).abs() < 1e-25,
            "Xe production from fission"
        );
    }

    // --- spectrum-weighted fission yields (issue #379) ---

    /// `U` fissions into `B` and `C`, with yields that swap between a thermal
    /// and a fast tabulated point. Deliberately asymmetric so a fold that
    /// silently picked one energy could not pass by coincidence.
    fn two_energy_fission_chain() -> HashMap<String, ChainNuclide> {
        let mut chain = HashMap::new();
        chain.insert(
            "U".to_string(),
            ChainNuclide {
                name: "U".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: vec![ChainReaction {
                    kind: "(n,fission)".to_string(),
                    target: None,
                    branching: 1.0,
                    q_value: None,
                }],
                decays: vec![],
                fission_yields: Some(Arc::new(FissionYieldSet::new(vec![
                    FissionYield {
                        energy: 0.0253,
                        products: vec![("B".to_string(), 1.6), ("C".to_string(), 0.4)],
                    },
                    FissionYield {
                        energy: 1.4e7,
                        products: vec![("B".to_string(), 0.6), ("C".to_string(), 1.4)],
                    },
                ]))),
                sources: Vec::new(),
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        for name in ["B", "C"] {
            chain.insert(name.to_string(), nuclide(name, None, vec![], vec![]));
        }
        chain
    }

    fn fission_rates(rate: f64) -> ReactionRates {
        let mut rates: ReactionRates = HashMap::new();
        rates.insert(
            "U".to_string(),
            HashMap::from([("(n,fission)".to_string(), rate)]),
        );
        rates
    }

    #[test]
    fn fold_at_a_tabulated_energy_reproduces_that_yield_vector_exactly() {
        // A delta spectrum on a tabulated point must select that point's
        // vector alone. With the delta at 0.0253 eV this is bit-identical to
        // the pre-#379 `yields.first()` behaviour, which also pins equivalence
        // to OpenMC's ConstantFissionYieldHelper default without a mode knob.
        let chain = two_energy_fission_chain();
        let names = vec!["B".to_string(), "C".to_string(), "U".to_string()];
        let rate = 1e-5;
        let rates = fission_rates(rate);

        for (k, (y_b, y_c)) in [(1.6, 0.4), (0.6, 1.4)].into_iter().enumerate() {
            let mut c = vec![0.0; 2];
            c[k] = 1.0;
            let fy = FissionYieldWeights::from([("U".to_string(), c)]);
            let m = build_matrix(&chain, &names, &rates, &fy, ChainParts::default()).unwrap();
            // Row-major 3x3: column 2 is U, rows 0 and 1 are B and C.
            assert_eq!(m[2], rate * y_b, "B at tabulated point {k}");
            assert_eq!(m[5], rate * y_c, "C at tabulated point {k}");
        }
    }

    #[test]
    fn fold_combines_tabulated_points_linearly() {
        // Half the fission rate at each tabulated point averages the vectors.
        let chain = two_energy_fission_chain();
        let names = vec!["B".to_string(), "C".to_string(), "U".to_string()];
        let rate = 1e-5;
        let fy = FissionYieldWeights::from([("U".to_string(), vec![0.5, 0.5])]);

        let m = build_matrix(
            &chain,
            &names,
            &fission_rates(rate),
            &fy,
            ChainParts::default(),
        )
        .unwrap();
        assert!((m[2] - rate * 1.1).abs() < 1e-25, "B = (1.6 + 0.6) / 2");
        assert!((m[5] - rate * 0.9).abs() < 1e-25, "C = (0.4 + 1.4) / 2");
    }

    #[test]
    fn fold_conserves_the_products_per_fission() {
        // Every tabulated vector here sums to 2.0, and the weights are a
        // partition of unity, so any spectrum must also produce 2.0.
        let chain = two_energy_fission_chain();
        let names = vec!["B".to_string(), "C".to_string(), "U".to_string()];
        let rate = 1e-5;

        for c in [vec![1.0, 0.0], vec![0.25, 0.75], vec![0.0, 1.0]] {
            let fy = FissionYieldWeights::from([("U".to_string(), c.clone())]);
            let m = build_matrix(
                &chain,
                &names,
                &fission_rates(rate),
                &fy,
                ChainParts::default(),
            )
            .unwrap();
            let produced = (m[2] + m[5]) / rate;
            assert!(
                (produced - 2.0).abs() < 1e-12,
                "weights {c:?} produced {produced} products per fission, not 2"
            );
        }
    }

    #[test]
    fn fold_keeps_one_matrix_entry_per_product() {
        // The triplet count is what the sparse solve pays for, so folding
        // before emitting must not multiply entries by the number of
        // contributing energies.
        let chain = two_energy_fission_chain();
        let names = vec!["B".to_string(), "C".to_string(), "U".to_string()];
        let rates = fission_rates(1e-5);

        let one = FissionYieldWeights::from([("U".to_string(), vec![1.0, 0.0])]);
        let both = FissionYieldWeights::from([("U".to_string(), vec![0.5, 0.5])]);
        let (t_one, _) =
            build_matrix_triplets(&chain, &names, &rates, &one, ChainParts::default()).unwrap();
        let (t_both, _) =
            build_matrix_triplets(&chain, &names, &rates, &both, ChainParts::default()).unwrap();
        assert_eq!(
            t_one.len(),
            t_both.len(),
            "a two-energy fold must not double the triplets"
        );
    }

    #[test]
    fn missing_weights_are_an_error_not_a_silent_fallback() {
        // Quietly falling back to one tabulated energy is exactly the defect
        // of issue #379, so the builder refuses instead.
        let chain = two_energy_fission_chain();
        let names = vec!["B".to_string(), "C".to_string(), "U".to_string()];
        let err = build_matrix(
            &chain,
            &names,
            &fission_rates(1e-5),
            &fy_weights(),
            ChainParts::default(),
        )
        .expect_err("a fissioning nuclide with no weights must be rejected");
        assert!(err.contains('U'), "error should name the nuclide: {err}");
    }

    #[test]
    fn a_zero_fission_rate_needs_no_weights() {
        // No fission means no fold to define, so the demand must not fire.
        let chain = two_energy_fission_chain();
        let names = vec!["B".to_string(), "C".to_string(), "U".to_string()];
        let m = build_matrix(
            &chain,
            &names,
            &fission_rates(0.0),
            &fy_weights(),
            ChainParts::default(),
        )
        .unwrap();
        assert!(m.iter().all(|v| *v == 0.0), "no rate, no matrix entries");
    }

    #[test]
    fn weights_longer_than_the_tabulation_are_an_error() {
        // Guards the positional contract between the weights and the yields.
        let chain = two_energy_fission_chain();
        let names = vec!["B".to_string(), "C".to_string(), "U".to_string()];
        let fy = FissionYieldWeights::from([("U".to_string(), vec![0.5, 0.25, 0.25])]);
        let err = build_matrix(
            &chain,
            &names,
            &fission_rates(1e-5),
            &fy,
            ChainParts::default(),
        )
        .expect_err("more weights than tabulated energies must be rejected");
        assert!(err.contains("tabulated"), "unexpected error: {err}");
    }

    #[test]
    fn test_decay_particle_products_known_types() {
        // Verify our light particle mappings
        assert_eq!(decay_particle_products("alpha"), &[("He4", 1.0)]);
        assert_eq!(decay_particle_products("p"), &[("H1", 1.0)]);
        assert_eq!(decay_particle_products("beta-,alpha"), &[("He4", 1.0)]);
        assert_eq!(decay_particle_products("ec/beta+,p"), &[("H1", 1.0)]);
        assert!(decay_particle_products("beta-").is_empty());
        assert!(decay_particle_products("IT").is_empty());
        assert!(decay_particle_products("SF").is_empty());
        assert!(decay_particle_products("unknown_mode").is_empty());
    }

    #[test]
    fn test_light_particle_products_known_types() {
        assert_eq!(light_particle_products("(n,p)"), &[("H1", 1.0)]);
        assert_eq!(light_particle_products("(n,a)"), &[("He4", 1.0)]);
        assert_eq!(light_particle_products("(n,d)"), &[("H2", 1.0)]);
        assert_eq!(light_particle_products("(n,t)"), &[("H3", 1.0)]);
        assert_eq!(light_particle_products("(n,2p)"), &[("H1", 2.0)]);
        assert_eq!(
            light_particle_products("(n,pa)"),
            &[("H1", 1.0), ("He4", 1.0)]
        );
        assert!(light_particle_products("(n,gamma)").is_empty());
        assert!(light_particle_products("(n,2n)").is_empty());
    }

    /// Issue #502: `Material.transmute()` returned two different inventories
    /// for identical inputs. The reaction kinds were grouped into a `HashMap`
    /// and both the diagonal loss sum and the order the matrix entries were
    /// emitted in followed its iteration order, which differs per map instance
    /// even inside one process.
    ///
    /// The rates are chosen so summation order is visible in the result: `1.0 +
    /// 1e-16` rounds back to `1.0`, so adding the small terms to the large one
    /// first loses them all, while adding them to each other first does not.
    #[test]
    fn the_matrix_is_bit_identical_across_builds() {
        let kinds = [
            ("(n,gamma)", 1.0),
            ("(n,2n)", 1.0e-16),
            ("(n,3n)", 1.0e-16),
            ("(n,p)", 1.0e-16),
            ("(n,a)", 1.0e-16),
            ("(n,d)", 1.0e-16),
        ];

        let mut chain = HashMap::new();
        chain.insert(
            "A".to_string(),
            nuclide(
                "A",
                None,
                vec![],
                kinds
                    .iter()
                    .map(|(kind, _)| ChainReaction {
                        kind: (*kind).to_string(),
                        target: Some("B".to_string()),
                        branching: 1.0,
                        q_value: None,
                    })
                    .collect(),
            ),
        );
        chain.insert("B".to_string(), nuclide("B", None, vec![], vec![]));

        let names = vec!["A".to_string(), "B".to_string()];
        let mut rates: ReactionRates = HashMap::new();
        rates.insert(
            "A".to_string(),
            kinds
                .iter()
                .map(|(kind, rate)| ((*kind).to_string(), *rate))
                .collect(),
        );

        // A fresh `grouped` map per call is the point, so this has to rebuild
        // rather than compare one result with itself.
        let first =
            build_matrix(&chain, &names, &rates, &fy_weights(), ChainParts::default()).unwrap();
        for i in 1..200 {
            let again =
                build_matrix(&chain, &names, &rates, &fy_weights(), ChainParts::default()).unwrap();
            assert_eq!(
                first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                again.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "build {i} differs from the first"
            );
        }
    }
}
