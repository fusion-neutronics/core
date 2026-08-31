// Reaction sampling methods for `Nuclide`.
//
// Split out of `nuclide.rs` into a child module; behavior-preserving move.
use super::*;

/// URR-adjusted reaction cross-sections plus the random number used, as
/// returned by `Nuclide::urr_adjusted_reaction_xs`. `urr_random` is
/// carried so call sites can keep their `debug_collision` log output
/// byte-identical.
struct UrrReactionXs {
    total: f64,
    capture: f64,
    scatter: f64,
    /// URR-sampled ELASTIC alone. `scatter` is this plus the inelastic that went
    /// into the URR total, and the two are NOT in the smooth elastic:inelastic
    /// ratio, because the probability table modifies elastic and leaves inelastic
    /// alone. Carried so the reaction split can use it directly (issue #372).
    elastic: f64,
    fission: f64,
    /// Only read by `debug_collision` logging at call sites.
    #[allow(dead_code)]
    urr_random: f64,
}

impl Nuclide {
    /// Sample URR-modified cross-sections at given energy.
    /// Returns (total, elastic, capture, fission) cross-sections after applying URR fluctuation factors.
    /// If energy is outside URR range or no URR data exists, returns the smooth cross-sections.
    ///
    /// Uses URR probability table sampling for proper resonance fluctuation treatment.
    /// The same random number should be used for all reactions in a single particle interaction
    /// to preserve correlations between reaction channels.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn sample_urr_xs<R: rand::Rng + ?Sized>(
        &self,
        energy: f64,
        temperature: &str,
        smooth_elastic: f64,
        smooth_capture: f64,
        smooth_fission: f64,
        smooth_inelastic: f64,
        rng: &mut R,
    ) -> (f64, f64, f64, f64) {
        let smooth_total = smooth_elastic + smooth_inelastic + smooth_capture + smooth_fission;

        // Get URR data for this temperature
        let Some(urr) = self.urr_for_temp(temperature) else {
            return (smooth_total, smooth_elastic, smooth_capture, smooth_fission);
        };

        // Check if energy is in URR bounds
        if !urr.energy_in_bounds(energy) {
            return (smooth_total, smooth_elastic, smooth_capture, smooth_fission);
        }

        // Sample random number for URR (should be consistent for all reactions in this interaction)
        let r = rng.random::<f64>();

        // Sample from URR probability tables
        // The smooth_absorption passed to URR should be capture + fission
        let smooth_absorption = smooth_capture + smooth_fission;
        // Pass smooth_ngamma as None here - this function doesn't have access to FastXSGrid.
        // For multiply_smooth=false nuclides, the ratio calculation may be slightly off
        // if other_abs is significant, but material.rs::apply_urr_to_total_xs handles
        // this properly with the xs_ngamma from FastXSGrid.
        let (urr_total, urr_elastic, urr_capture, urr_fission, _) = urr.sample(
            energy,
            r,
            smooth_elastic,
            smooth_absorption,
            smooth_fission,
            smooth_inelastic,
            None,
        );

        (urr_total, urr_elastic, urr_capture, urr_fission)
    }

    /// Apply the URR probability-table adjustment to smooth reaction
    /// cross-sections, preserving the correlation with
    /// distance-to-collision sampling: `urr_random` (the per-collision
    /// random cached on the particle) is used when present; a fresh
    /// random is drawn otherwise. Returns `None` when URR is absent for
    /// this temperature or `energy` is outside the table bounds; callers
    /// then keep the smooth values.
    ///
    /// `xs_elastic` is a closure so the elastic lookup (a grid
    /// interpolation on the fast path) is only paid inside the URR
    /// window.
    ///
    /// Shared by `sample_reaction_type_fast`, both paths of
    /// `sample_reaction_type`, and `collision_xs`. Distinct from the
    /// public [`Self::sample_urr_xs`], which always draws a fresh random
    /// and serves the material-level total-XS path.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn urr_adjusted_reaction_xs<R: rand::Rng + ?Sized>(
        &self,
        temp_idx: usize,
        energy: f64,
        xs_elastic: impl FnOnce() -> f64,
        xs_scattering: f64,
        xs_absorption: f64,
        xs_fission: f64,
        urr_random: Option<f64>,
        rng: &mut R,
    ) -> Option<UrrReactionXs> {
        if !self.urr_present {
            return None;
        }
        let Some(Some(urr)) = self.urr_data.get(temp_idx) else {
            return None;
        };
        if !urr.energy_in_bounds(energy) {
            return None;
        }

        // Get elastic XS separately for URR modification
        let xs_elastic = xs_elastic();

        // Inelastic is total scattering minus elastic (smooth values)
        let xs_inelastic = (xs_scattering - xs_elastic).max(0.0);

        // `xs_absorption` is the grid's absorption PARTIAL, which already excludes
        // fission: `FastXSGrid::lookup` returns four partials that sum to the
        // total (elastic + inelastic inside `xs_scattering`, then absorption and
        // fission separately), and the analog reaction split adds all four.
        // Subtracting fission here as well drove capture to ZERO for every nuclide
        // whose in-band fission exceeds its capture, i.e. every fissile one --
        // U235 at 10 keV has absorption 1.06 b against fission 2.91 b, so the
        // whole in-band capture was lost and the URR-adjusted total came out ~7%
        // low (issue #154). W184, the only URR fixture, has no fission at all,
        // which is why it never showed it.
        let xs_capture = xs_absorption;

        // The per-collision base seed correlates the URR band with distance
        // sampling; derive this nuclide's independent band from it so a
        // material's isotopes sample uncorrelated resonance structure and the
        // *struck* nuclide's reaction reuses the exact band its flight used
        // (issue #204).
        let base = urr_random.unwrap_or_else(|| rng.random::<f64>());
        let r = crate::urr::urr_nuclide_random(base, self.urr_stream_key());

        // Sample URR-modified cross-sections
        // urr.sample() now computes total = elastic + inelastic + capture + fission
        // where inelastic is zeroed when inelastic_flag <= 0 (per ENDF convention), else uses smooth value
        let smooth_absorption = xs_capture + xs_fission;
        let (urr_total, urr_elastic, urr_capture, urr_fission, _) = urr.sample(
            energy,
            r,
            xs_elastic,
            smooth_absorption,
            xs_fission,
            xs_inelastic,
            None,
        );

        Some(UrrReactionXs {
            total: urr_total,
            capture: urr_capture,
            // Scattering = total - absorption = elastic + inelastic
            // (urr_total already has correct inelastic baked in)
            scatter: (urr_total - urr_capture - urr_fission).max(0.0),
            elastic: urr_elastic,
            fission: urr_fission,
            urr_random: r,
        })
    }

    /// Sample the top-level reaction type (fission, absorption, elastic, inelastic, other) at a given energy and temperature
    /// This version includes auto-loading if data is not available.
    pub fn sample_reaction<R: rand::Rng + ?Sized>(
        &mut self,
        energy: f64,
        temperature: &str,
        rng: &mut R,
    ) -> Option<&Reaction> {
        // Auto-load data if not available
        if self.reactions.is_empty() && self.name.is_some() {
            if let Some(name) = &self.name.clone() {
                if self.auto_load_from_config(name, Some(temperature)).is_err() {
                    println!("[sample_reaction] Failed to auto-load data for nuclide '{name}'");
                    return None;
                }
            }
        }

        let temp_reactions = self.reactions_for_temp(temperature)?;

        // Define MTs for each event type
        let total_mt = 1;
        let fission_mt = 18;
        let absorption_mt = 101;
        let elastic_mt = 2;
        let inelastic_mt = 4;

        // Helper to get xs for a given MT using Reaction::cross_section_at
        let get_xs = |mt: i32| -> f64 {
            temp_reactions
                .get(&mt)
                .and_then(|reaction| reaction.cross_section_at(energy))
                .unwrap_or(0.0)
        };

        let total_xs = get_xs(total_mt);
        if total_xs <= 0.0 {
            return None;
        }

        let xi = rng.random_range(0.0..total_xs);
        let mut accum = 0.0;

        // Absorption

        let xs_absorption = get_xs(absorption_mt);
        accum += xs_absorption;
        if xi < accum && xs_absorption > 0.0 {
            return temp_reactions.get(&absorption_mt).map(|r| r.as_ref());
        }

        // Elastic
        let xs_elastic = get_xs(elastic_mt);
        accum += xs_elastic;
        if xi < accum && xs_elastic > 0.0 {
            return temp_reactions.get(&elastic_mt).map(|r| r.as_ref());
        }

        // Fission (only if nuclide is fissionable, checked last)
        let xs_fission = if self.fissionable {
            get_xs(fission_mt)
        } else {
            0.0
        };
        accum += xs_fission;
        if xi < accum && xs_fission > 0.0 {
            return temp_reactions.get(&fission_mt).map(|r| r.as_ref());
        }

        // inelastic selection as fallback
        temp_reactions.get(&inelastic_mt).map(|r| r.as_ref())
    }

    /// Sample the top-level reaction type using fast pre-computed cross-sections.
    /// This is the optimized version that uses logarithmic grid lookup instead of binary search.
    /// Returns a ReactionType enum indicating which category of reaction was sampled.
    /// For scattering, call sample_scattering_constituent to get the specific reaction.
    ///
    /// When energy is in the unresolved resonance range (URR) and URR probability tables are
    /// available, cross-sections are sampled from the probability tables to account for
    /// resonance fluctuations (see ENDF-102, Section 2.3.1: Unresolved Resonance Parameters).
    ///
    /// If `urr_random` is Some, uses that random number for URR sampling to maintain correlation
    /// with distance-to-collision sampling. If None, samples a new URR random if needed.
    #[inline]
    pub fn sample_reaction_type_fast<R: rand::Rng + ?Sized>(
        &self,
        energy: f64,
        temperature: &str,
        urr_random: Option<f64>,
        rng: &mut R,
    ) -> Option<ReactionType> {
        // Use fast XS lookup if available
        let temp_idx = self.get_temp_idx(temperature)?;
        let fast_grid = self.fast_xs.get(temp_idx)?;

        let (mut total_xs, mut xs_absorption, xs_scattering, mut xs_fission) =
            fast_grid.lookup(energy);

        if total_xs <= 0.0 {
            return None;
        }

        // Check if we're in URR range and need to apply probability table sampling
        let mut xs_scattering_final = xs_scattering;
        if let Some(urr_xs) = self.urr_adjusted_reaction_xs(
            temp_idx,
            energy,
            || {
                let (i_grid, f) = fast_grid.lookup_grid_index(energy);
                fast_grid
                    .elastic_idx
                    .map(|idx| fast_grid.scatter_xs_interp(i_grid, f, idx))
                    .unwrap_or(0.0)
            },
            xs_scattering,
            xs_absorption,
            xs_fission,
            urr_random,
            rng,
        ) {
            xs_scattering_final = urr_xs.scatter;
            xs_absorption = urr_xs.capture;
            xs_fission = urr_xs.fission;
            total_xs = urr_xs.total;

            // Debug collision logging: URR reaction sampling (fast path)
            #[cfg(feature = "debug_collision")]
            if is_debug_nuclide() && energy_in_debug_range_nuc(energy) {
                let count = DEBUG_NUCLIDE_COUNT.fetch_add(1, Ordering::Relaxed);
                if count < DEBUG_NUCLIDE_MAX {
                    let p_abs = if total_xs > 0.0 {
                        xs_absorption / total_xs
                    } else {
                        0.0
                    };
                    let p_scat = if total_xs > 0.0 {
                        xs_scattering_final / total_xs
                    } else {
                        0.0
                    };
                    eprintln!(
                        "[NUC_RXN_FAST] E={:.2}eV urr_r={:.6} xs_abs={:.4} xs_scat={:.4} total={:.4} P(abs)={:.4} P(scat)={:.4}",
                        energy, urr_xs.urr_random, xs_absorption, xs_scattering_final, total_xs, p_abs, p_scat
                    );
                }
            }
        }

        if total_xs <= 0.0 {
            return None;
        }

        let xi = rng.random_range(0.0..total_xs);

        // Debug collision logging: reaction sampling random number (fast path)
        #[cfg(feature = "debug_collision")]
        if is_debug_nuclide() && energy_in_debug_range_nuc(energy) {
            let count = DEBUG_NUCLIDE_COUNT.load(Ordering::Relaxed);
            if count < DEBUG_NUCLIDE_MAX {
                eprintln!(
                    "[NUC_SAMPLE_FAST] E={:.2}eV xi={:.6} total={:.4} (abs={:.4} scat={:.4} fis={:.4})",
                    energy, xi, total_xs, xs_absorption, xs_scattering_final, xs_fission
                );
            }
        }

        let mut accum = 0.0;

        // Absorption
        accum += xs_absorption;
        if xi < accum && xs_absorption > 0.0 {
            return Some(ReactionType::Absorption);
        }

        // Scattering
        accum += xs_scattering_final;
        if xi < accum && xs_scattering_final > 0.0 {
            return Some(ReactionType::Scattering);
        }

        // Fission
        accum += xs_fission;
        if xi < accum && xs_fission > 0.0 {
            return Some(ReactionType::Fission);
        }

        // Fallback to scattering
        Some(ReactionType::Scattering)
    }

    /// Sample the top-level reaction type without auto-loading (requires data to be pre-loaded)
    /// This version is used when the nuclide is stored in shared/immutable contexts like Arc<Nuclide>.
    /// Returns ReactionType enum - for scattering, use sample_scattering_constituent to get the specific reaction.
    ///
    /// When energy is in the unresolved resonance range (URR) and URR probability tables are
    /// available, cross-sections are sampled from the probability tables to account for
    /// resonance fluctuations.
    ///
    /// If `urr_random` is Some, uses that random number for URR sampling to maintain correlation
    /// with distance-to-collision sampling. If None, samples a new URR random if needed.
    #[inline]
    pub fn sample_reaction_type<R: rand::Rng + ?Sized>(
        &self,
        energy: f64,
        temperature: &str,
        urr_random: Option<f64>,
        rng: &mut R,
    ) -> Option<ReactionType> {
        // Get temperature index for O(1) lookup
        let temp_idx = self.get_temp_idx(temperature)?;

        // Use fast XS lookup if available (log-binned energy grid for O(1) access)
        if let Some(fast_grid) = self.fast_xs.get(temp_idx) {
            let (mut total_xs, mut xs_absorption, xs_scattering, mut xs_fission) =
                fast_grid.lookup(energy);

            if total_xs <= 0.0 {
                return None;
            }

            // Check if we're in URR range and need to apply probability table sampling
            let mut xs_scattering_final = xs_scattering;
            if let Some(urr_xs) = self.urr_adjusted_reaction_xs(
                temp_idx,
                energy,
                || {
                    // Elastic is MT 2; use the cached index into scatter_mt_xs.
                    let (i_grid, f) = fast_grid.lookup_grid_index(energy);
                    fast_grid
                        .elastic_idx
                        .map(|idx| fast_grid.scatter_xs_interp(i_grid, f, idx))
                        .unwrap_or(0.0)
                },
                xs_scattering,
                xs_absorption,
                xs_fission,
                urr_random,
                rng,
            ) {
                xs_scattering_final = urr_xs.scatter;
                xs_absorption = urr_xs.capture;
                xs_fission = urr_xs.fission;
                total_xs = urr_xs.total;

                // Debug collision logging: URR reaction sampling
                #[cfg(feature = "debug_collision")]
                if is_debug_nuclide() && energy_in_debug_range_nuc(energy) {
                    let count = DEBUG_NUCLIDE_COUNT.fetch_add(1, Ordering::Relaxed);
                    if count < DEBUG_NUCLIDE_MAX {
                        let p_abs = if total_xs > 0.0 {
                            xs_absorption / total_xs
                        } else {
                            0.0
                        };
                        let p_scat = if total_xs > 0.0 {
                            xs_scattering_final / total_xs
                        } else {
                            0.0
                        };
                        eprintln!(
                            "[NUC_RXN] E={:.2}eV urr_r={:.6} xs_abs={:.4} xs_scat={:.4} xs_fis={:.4} total={:.4} P(abs)={:.4} P(scat)={:.4}",
                            energy, urr_xs.urr_random, xs_absorption, xs_scattering_final, xs_fission, total_xs, p_abs, p_scat
                        );
                    }
                }
            }

            if total_xs <= 0.0 {
                return None;
            }

            let xi = rng.random_range(0.0..total_xs);

            // Debug collision logging: reaction sampling random number and probabilities
            #[cfg(feature = "debug_collision")]
            if is_debug_nuclide() && energy_in_debug_range_nuc(energy) {
                let count = DEBUG_NUCLIDE_COUNT.load(Ordering::Relaxed);
                if count < DEBUG_NUCLIDE_MAX {
                    eprintln!(
                        "[NUC_SAMPLE] E={:.2}eV xi={:.6} total_xs={:.4} (abs={:.4} scat={:.4} fis={:.4})",
                        energy, xi, total_xs, xs_absorption, xs_scattering_final, xs_fission
                    );
                }
            }
            let mut accum = 0.0;

            // Absorption (capture, not fission)
            accum += xs_absorption;
            if xi < accum && xs_absorption > 0.0 {
                return Some(ReactionType::Absorption);
            }

            // Scattering (elastic + inelastic)
            accum += xs_scattering_final;
            if xi < accum && xs_scattering_final > 0.0 {
                return Some(ReactionType::Scattering);
            }

            // Fission
            accum += xs_fission;
            if xi < accum && xs_fission > 0.0 {
                return Some(ReactionType::Fission);
            }

            // Fallback to scattering
            return Some(ReactionType::Scattering);
        }

        // Fallback to slow path if fast_xs not initialized
        let temp_reactions = self.reactions.get(temp_idx)?;

        // Helper to get xs for a given MT using Reaction::cross_section_at
        let get_xs = |mt: i32| -> f64 {
            temp_reactions
                .get(&mt)
                .and_then(|reaction| reaction.cross_section_at(energy))
                .unwrap_or(0.0)
        };

        let mut total_xs = get_xs(1); // MT 1 = total
        if total_xs <= 0.0 {
            return None;
        }

        // Compute scattering XS by summing all scattering reactions
        let mut xs_scattering = 0.0;
        for (&mt, reaction) in temp_reactions.iter() {
            if is_scattering_mt(mt) && !reaction.redundant {
                if let Some(xs) = reaction.cross_section_at(energy) {
                    xs_scattering += xs;
                }
            }
        }

        let mut xs_absorption = get_xs(101);
        let mut xs_fission = if self.fissionable { get_xs(18) } else { 0.0 };

        // Apply URR if in range (slow path)
        if let Some(urr_xs) = self.urr_adjusted_reaction_xs(
            temp_idx,
            energy,
            || get_xs(2),
            xs_scattering,
            xs_absorption,
            xs_fission,
            urr_random,
            rng,
        ) {
            xs_scattering = urr_xs.scatter;
            xs_absorption = urr_xs.capture;
            xs_fission = urr_xs.fission;
            total_xs = urr_xs.total;
        }

        if total_xs <= 0.0 {
            return None;
        }

        let xi = rng.random_range(0.0..total_xs);
        let mut accum = 0.0;

        // Absorption
        accum += xs_absorption;
        if xi < accum && xs_absorption > 0.0 {
            return Some(ReactionType::Absorption);
        }

        // Scattering
        accum += xs_scattering;
        if xi < accum && xs_scattering > 0.0 {
            return Some(ReactionType::Scattering);
        }

        // Fission
        accum += xs_fission;
        if xi < accum && xs_fission > 0.0 {
            return Some(ReactionType::Fission);
        }

        // Fallback to scattering if nothing else was sampled
        Some(ReactionType::Scattering)
    }

    /// Cross-section breakdown at a collision site, with the same URR
    /// probability-table adjustment as [`Self::sample_reaction_type`] but
    /// without sampling a reaction channel.
    ///
    /// Used by survival biasing (implicit capture): the transport loop
    /// needs total / scattering / fission at the collision energy to
    /// compute the survival factor and the expected fission progeny.
    ///
    /// `urr_random` should be the per-collision URR random cached on the
    /// particle so the breakdown stays correlated with the
    /// distance-to-collision sample; a fresh random is drawn only when it
    /// is `None` while the energy is in URR range (mirroring
    /// `sample_reaction_type`).
    ///
    /// Returns `None` when no data exists for `temperature` or the total
    /// cross-section vanishes.
    pub fn collision_xs<R: rand::Rng + ?Sized>(
        &self,
        energy: f64,
        temperature: &str,
        urr_random: Option<f64>,
        rng: &mut R,
    ) -> Option<CollisionXs> {
        let temp_idx = self.get_temp_idx(temperature)?;

        // Fast path: pre-computed log-binned grid (what transport uses).
        if let Some(fast_grid) = self.fast_xs.get(temp_idx) {
            let (mut total_xs, xs_absorption, xs_scattering, mut xs_fission) =
                fast_grid.lookup(energy);

            if total_xs <= 0.0 {
                return None;
            }

            let mut xs_scattering_final = xs_scattering;
            if let Some(urr_xs) = self.urr_adjusted_reaction_xs(
                temp_idx,
                energy,
                || {
                    let (i_grid, f) = fast_grid.lookup_grid_index(energy);
                    fast_grid
                        .elastic_idx
                        .map(|idx| fast_grid.scatter_xs_interp(i_grid, f, idx))
                        .unwrap_or(0.0)
                },
                xs_scattering,
                xs_absorption,
                xs_fission,
                urr_random,
                rng,
            ) {
                xs_scattering_final = urr_xs.scatter;
                xs_fission = urr_xs.fission;
                total_xs = urr_xs.total;
            }

            if total_xs <= 0.0 {
                return None;
            }

            return Some(CollisionXs {
                total: total_xs,
                scatter: xs_scattering_final,
                fission: xs_fission,
            });
        }

        // Slow fallback: direct reaction-map lookups, mirroring
        // `sample_reaction_type`.
        let temp_reactions = self.reactions.get(temp_idx)?;
        let get_xs = |mt: i32| -> f64 {
            temp_reactions
                .get(&mt)
                .and_then(|reaction| reaction.cross_section_at(energy))
                .unwrap_or(0.0)
        };

        let mut total_xs = get_xs(1);
        if total_xs <= 0.0 {
            return None;
        }

        let mut xs_scattering = 0.0;
        for (&mt, reaction) in temp_reactions.iter() {
            if is_scattering_mt(mt) && !reaction.redundant {
                if let Some(xs) = reaction.cross_section_at(energy) {
                    xs_scattering += xs;
                }
            }
        }

        let xs_absorption = get_xs(101);
        let mut xs_fission = if self.fissionable { get_xs(18) } else { 0.0 };

        if let Some(urr_xs) = self.urr_adjusted_reaction_xs(
            temp_idx,
            energy,
            || get_xs(2),
            xs_scattering,
            xs_absorption,
            xs_fission,
            urr_random,
            rng,
        ) {
            xs_scattering = urr_xs.scatter;
            xs_fission = urr_xs.fission;
            total_xs = urr_xs.total;
        }

        if total_xs <= 0.0 {
            return None;
        }

        Some(CollisionXs {
            total: total_xs,
            scatter: xs_scattering,
            fission: xs_fission,
        })
    }

    /// Reaction-channel partial cross-sections `(sigma_e, sigma_a, sigma_i,
    /// sigma_f)` for the analog reaction-type split (issue #111), computed by
    /// exactly mirroring [`Self::sample_reaction_type`]'s cross-section lookup
    /// and URR adjustment, then splitting the scattering bucket into elastic
    /// (MT 2) and inelastic by the smooth elastic/scatter ratio. The four
    /// partials sum to the (URR-adjusted) total, so a single uniform `xi2`
    /// partitions the channels with the same marginal probabilities the
    /// two-step `sample_reaction_type` + `sample_scattering_constituent`
    /// selection produces (OpenMC parity preserved; only the draw's cumulative
    /// ordering changes).
    ///
    /// Fast path only: returns `None` when `temperature` has no `fast_xs` grid
    /// or the total cross-section vanishes. The transport caller falls back to
    /// the legacy `sample_reaction_type` (which carries its own slow path) on
    /// `None`, so `elastic_reaction` / `sample_inelastic_scatter_reaction` --
    /// also fast-path only -- are reached only when this returned `Some` (i.e.
    /// `fast_xs` is present), keeping the three consistent.
    ///
    /// `urr_random` is the per-collision random cached on the particle, used
    /// (without drawing) to keep the URR correlation; a fresh random is drawn
    /// only when it is `None` while in URR range, mirroring
    /// `sample_reaction_type`.
    pub fn reaction_partials<R: rand::Rng + ?Sized>(
        &self,
        energy: f64,
        temperature: &str,
        urr_random: Option<f64>,
        rng: &mut R,
    ) -> Option<ReactionPartials> {
        let temp_idx = self.get_temp_idx(temperature)?;
        let fast_grid = self.fast_xs.get(temp_idx)?;

        let (mut total_xs, mut xs_absorption, xs_scattering, mut xs_fission) =
            fast_grid.lookup(energy);
        if total_xs <= 0.0 {
            return None;
        }

        // Smooth elastic (MT 2) via the cached scatter-table column -- the same
        // value `sample_scattering_constituent` walks for the constituent split.
        let (i_grid, f) = fast_grid.lookup_grid_index(energy);
        let smooth_elastic = fast_grid
            .elastic_idx
            .map(|idx| fast_grid.scatter_xs_interp(i_grid, f, idx))
            .unwrap_or(0.0);
        let smooth_scatter = xs_scattering;

        let mut scatter = xs_scattering;
        // `None` in band means the elastic:inelastic split below uses the smooth
        // ratio, which is exact off the probability table.
        let mut urr_elastic: Option<f64> = None;
        if let Some(urr_xs) = self.urr_adjusted_reaction_xs(
            temp_idx,
            energy,
            || smooth_elastic,
            xs_scattering,
            xs_absorption,
            xs_fission,
            urr_random,
            rng,
        ) {
            scatter = urr_xs.scatter;
            xs_absorption = urr_xs.capture;
            xs_fission = urr_xs.fission;
            total_xs = urr_xs.total;
            urr_elastic = Some(urr_xs.elastic);
        }
        if total_xs <= 0.0 {
            return None;
        }

        // Split the scatter bucket into elastic + inelastic.
        //
        // In a URR band the split is already KNOWN: the probability table scales
        // ELASTIC and leaves inelastic alone, so `scatter` is exactly
        // `urr_elastic + inelastic` and the elastic partial is `urr_elastic`
        // itself. Re-deriving it from the SMOOTH elastic:scatter ratio, which is
        // what this did, silently discards the band: at 1 MeV in Fe58's table the
        // smooth ratio is 0.843 while the true per-band elastic fraction runs
        // 0.709 (low band) to 0.902 (high band), so a low band came out 19% too
        // elastic and 46% short on inelastic. Averaged over bands the elastic mean
        // survives (the tables preserve it) but the per-band correlation with the
        // flux does not, which is the self-shielding the tables exist to model.
        // Issue #372: this is what put every URR-bearing nuclide at the top of the
        // V&V outlier list (Fe58 chi2/dof 20, Mn55 11, Ni62 4.7) while every
        // non-URR nuclide sat at ~1.1.
        //
        // Out of band, fall back to the smooth ratio as before.
        let sigma_e = match urr_elastic {
            Some(e) => e.min(scatter),
            None if smooth_scatter > 0.0 => scatter * (smooth_elastic / smooth_scatter),
            None => 0.0,
        };
        let sigma_i = (scatter - sigma_e).max(0.0);
        Some(ReactionPartials {
            sigma_e,
            sigma_a: xs_absorption,
            sigma_i,
            sigma_f: xs_fission,
        })
    }

    /// The elastic (MT 2) reaction, if present, for the analog reaction-type
    /// split (issue #111): once `xi2` selects the elastic channel directly, the
    /// caller needs the MT 2 `Reaction` to fetch its angular table. Returns the
    /// same `&Reaction` `sample_scattering_constituent` would yield for MT 2.
    pub fn elastic_reaction(&self, temperature: &str) -> Option<&Reaction> {
        let temp_idx = self.get_temp_idx(temperature)?;
        let fast_grid = self.fast_xs.get(temp_idx)?;
        fast_grid.elastic_reaction()
    }

    /// Select a non-elastic scattering constituent proportional to its smooth
    /// cross-section, driven by a pre-drawn PCG uniform `xi_mt` in `(0, 1]`
    /// (issue #111). Used by the analog inelastic branch after `xi2` has decided
    /// elastic-vs-inelastic, replacing the elastic/inelastic part of
    /// `sample_scattering_constituent` on the shared PCG stream.
    pub fn sample_inelastic_scatter_reaction(
        &self,
        energy: f64,
        temperature: &str,
        xi_mt: f64,
    ) -> Option<&Reaction> {
        let temp_idx = self.get_temp_idx(temperature)?;
        let fast_grid = self.fast_xs.get(temp_idx)?;
        fast_grid.sample_inelastic_scatter_reaction(energy, xi_mt)
    }

    /// Sample a specific scattering constituent reaction from all available scattering MTs.
    /// This samples from MT 2 (elastic), MT 50-91 (inelastic constituents), and other scattering MTs.
    /// Note: NEVER returns MT 4 (inelastic) as it's a synthetic reaction.
    ///
    /// # Arguments
    /// * `energy` - Neutron energy in eV
    /// * `temperature` - Temperature string in Kelvin (e.g., "294")
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// * `&Reaction` for the sampled constituent scattering reaction (MT 2, 50-91, 16, 17, etc.)
    ///
    /// # Panics
    /// * If no scattering constituent reactions are available
    /// * If sampling logic fails despite having valid reactions and cross sections
    #[inline]
    pub fn sample_scattering_constituent<R: rand::Rng + ?Sized>(
        &self,
        energy: f64,
        temperature: &str,
        rng: &mut R,
    ) -> &Reaction {
        // Get temperature index for O(1) lookup
        let temp_idx = self
            .get_temp_idx(temperature)
            .expect("[sample_scattering_constituent] Temperature not found");

        // Fast path: use pre-computed scattering XS data with cached reaction pointers
        // This avoids HashMap lookup entirely - returns &Reaction directly
        if let Some(fast_grid) = self.fast_xs.get(temp_idx) {
            if !fast_grid.scatter_mt_xs.is_empty() {
                if let Some(reaction) = fast_grid.sample_scatter_reaction(energy, rng) {
                    return reaction;
                }
            }
        }

        // Slow path fallback: needs temp_reactions for indexed lookup
        let temp_reactions = self
            .reactions
            .get(temp_idx)
            .expect("[sample_scattering_constituent] No reaction data for temperature");

        // Slow path: two-pass approach with binary search per reaction
        // Pass 1: Compute total scattering cross section
        let mut total_scattering_xs = 0.0;
        for (&mt, reaction) in temp_reactions.iter() {
            if mt == 4 || mt == 1 || mt == 18 || mt == 101 || mt == 1001 {
                continue;
            }
            // Skip redundant reactions for transport physics
            if reaction.redundant {
                continue;
            }
            if is_scattering_mt(mt) {
                if let Some(xs) = reaction.cross_section_at(energy) {
                    if xs > 0.0 {
                        total_scattering_xs += xs;
                    }
                }
            }
        }

        if total_scattering_xs <= 0.0 {
            panic!("sample_scattering_constituent: No scattering reactions at energy {energy} eV");
        }

        // Pass 2: Sample reaction
        let xi = rng.random_range(0.0..total_scattering_xs);
        let mut accum = 0.0;
        for (&mt, reaction) in temp_reactions.iter() {
            if mt == 4 || mt == 1 || mt == 18 || mt == 101 || mt == 1001 {
                continue;
            }
            // Skip redundant reactions for transport physics
            if reaction.redundant {
                continue;
            }
            if is_scattering_mt(mt) {
                if let Some(xs) = reaction.cross_section_at(energy) {
                    if xs > 0.0 {
                        accum += xs;
                        if xi < accum {
                            return reaction;
                        }
                    }
                }
            }
        }

        panic!("sample_scattering_constituent: sampling failed");
    }

    /// The nuclide's delayed-neutron groups: their yields `nu_d,g(E)` and the
    /// yield-weighted fold of their spectra, resolved on first call and cached
    /// (issue #364). `None` means the evaluation carries no delayed data.
    ///
    /// Delayed neutrons come from the fission products' decay, so they belong to
    /// fission as a whole rather than to a chance-fission channel, and ENDF hangs
    /// them off total fission (MT 18). This therefore takes the first fission
    /// reaction that carries any, independently of which channel a given event
    /// sampled -- unlike the prompt spectrum, which is per channel.
    ///
    /// `temperature` only selects which grid to read the reactions from; the
    /// products, and so the result, are the same at every temperature.
    pub fn delayed_neutrons(
        &self,
        temperature: &str,
    ) -> Option<&crate::delayed_neutrons::DelayedNeutronData> {
        self.delayed_neutron_cache.get_or_build(|| {
            let fast_grid = self.fast_xs.get(self.get_temp_idx(temperature)?)?;
            fast_grid.fission_mt_reactions.iter().find_map(|r| {
                crate::delayed_neutrons::DelayedNeutronData::from_products(&r.products)
            })
        })
    }

    /// Sample which fission reaction to use, proportional to the non-redundant
    /// partial fission cross sections (MT 18, 19, 20, 21, 38), driven by a PCG
    /// uniform in `(0, 1]` that `draw_xi` supplies (issue #418).
    ///
    /// `draw_xi` is called at most once, and only when there is more than one
    /// channel to choose between. See [`FastXSGrid::sample_fission_reaction`] for
    /// why the draw has to be skipped rather than taken and discarded.
    ///
    /// # Arguments
    /// * `energy` - Neutron energy in eV
    /// * `temperature` - Temperature string in Kelvin (e.g., "294")
    /// * `draw_xi` - Supplies one uniform in `(0, 1]` from the shared PCG stream
    ///
    /// # Returns
    /// * `Option<&Reaction>` for the sampled fission reaction, or None if no fission reactions available
    #[inline]
    pub fn sample_fission_reaction(
        &self,
        energy: f64,
        temperature: &str,
        draw_xi: impl FnOnce() -> f64,
    ) -> Option<&Reaction> {
        if !self.fissionable {
            return None;
        }

        // Get temperature index for O(1) lookup
        let temp_idx = self.get_temp_idx(temperature)?;

        // Fast path: use pre-computed fission XS data with cached reaction pointers
        if let Some(fast_grid) = self.fast_xs.get(temp_idx) {
            if !fast_grid.fission_mt_xs.is_empty() {
                return fast_grid.sample_fission_reaction(energy, draw_xi);
            }
        }

        // Slow path fallback: needs temp_reactions for indexed lookup
        let temp_reactions = self.reactions.get(temp_idx)?;

        // Collect non-redundant fission reactions, ordered by MT. `reactions` is a
        // HashMap, so without the sort the cumulative walk below would visit the
        // channels in an order that varies between processes, and a fixed seed
        // would not reproduce a fixed answer.
        let mut fission_rxns: Vec<(i32, &Reaction, f64)> = Vec::new();
        for (&mt, reaction) in temp_reactions.iter() {
            if reaction.redundant {
                continue;
            }
            if is_fission_mt(mt) {
                if let Some(xs) = reaction.cross_section_at(energy) {
                    if xs > 0.0 {
                        fission_rxns.push((mt, reaction, xs));
                    }
                }
            }
        }
        fission_rxns.sort_unstable_by_key(|&(mt, _, _)| mt);

        if fission_rxns.is_empty() {
            return None;
        }

        // One channel: no choice to make, so no draw.
        if fission_rxns.len() == 1 {
            return Some(fission_rxns[0].1);
        }

        // Sample proportionally among fission reactions
        let total_xs: f64 = fission_rxns.iter().map(|&(_, _, xs)| xs).sum();
        let target = draw_xi() * total_xs;
        let mut accum = 0.0;
        for &(_, reaction, xs) in &fission_rxns {
            accum += xs;
            if accum >= target {
                return Some(reaction);
            }
        }

        // Numerical fallback: the last channel in the walk.
        fission_rxns.last().map(|&(_, reaction, _)| reaction)
    }

    /// Sample a specific inelastic reaction from the constituent MT 50-91 reactions that make up MT 4.
    /// This provides more detailed physics than just sampling MT 4 directly.
    ///
    /// # Arguments
    /// * `energy` - Neutron energy in eV
    /// * `temperature` - Temperature string in Kelvin (e.g., "294")
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// * `&Reaction` for the sampled constituent inelastic reaction (MT 50-91)
    ///
    /// # Panics
    /// * If no inelastic constituent reactions (MT 50-91) are available
    /// * If sampling logic fails despite having valid reactions and cross sections
    pub fn sample_inelastic_constituent<R: rand::Rng + ?Sized>(
        &self,
        energy: f64,
        temperature: &str,
        rng: &mut R,
    ) -> &Reaction {
        // Get temperature index for O(1) lookup
        let temp_idx = self
            .get_temp_idx(temperature)
            .expect("[sample_inelastic_constituent] Temperature not found");
        let temp_reactions = self
            .reactions
            .get(temp_idx)
            .expect("[sample_inelastic_constituent] No reaction data for temperature");

        // MT 4 is composed of MT 50-91 (inelastic scattering to discrete levels)
        let inelastic_constituent_mts: Vec<i32> = (50..92).collect();

        // Filter to only the MTs that are actually available in this nuclide
        let available_inelastic_mts: Vec<i32> = inelastic_constituent_mts
            .into_iter()
            .filter(|&mt| temp_reactions.contains_key(&mt))
            .collect();

        if available_inelastic_mts.is_empty() {
            panic!("sample_inelastic_constituent: No inelastic constituent reactions (MT 50-91) available in this nuclide at temperature '{temperature}'. This indicates missing nuclear data or incorrect MT 4 sampling.");
        }

        // Helper to get cross section for a given MT
        let get_xs = |mt: i32| -> f64 {
            temp_reactions
                .get(&mt)
                .and_then(|reaction| reaction.cross_section_at(energy))
                .unwrap_or(0.0)
        };

        // Calculate total cross section for all available inelastic constituents
        let total_inelastic_xs: f64 = available_inelastic_mts.iter().map(|&mt| get_xs(mt)).sum();

        if total_inelastic_xs <= 0.0 {
            panic!("sample_inelastic_constituent: Total inelastic cross section is zero at energy {energy} eV for temperature '{temperature}'. All constituent reactions have zero cross section.");
        }

        // Sample which specific inelastic reaction occurs
        let xi = rng.random_range(0.0..total_inelastic_xs);
        let mut accum = 0.0;

        for &mt in &available_inelastic_mts {
            let xs = get_xs(mt);
            accum += xs;
            if xi < accum && xs > 0.0 {
                return temp_reactions.get(&mt).expect(
                    "sample_inelastic_constituent: MT not found in temp_reactions after filtering.",
                );
            }
        }

        // This should never be reached due to the sampling logic above
        panic!("sample_inelastic_constituent: Failed to sample any inelastic reaction despite having available MTs and positive total cross section. This indicates a bug in the sampling logic.");
    }

    /// Sample a specific absorption constituent reaction (MTs that make up MT 101) at a given energy and temperature.
    /// Panics if no constituent reactions are available or total cross section is zero.
    #[inline]
    pub fn sample_absorption_constituent<R: rand::Rng + ?Sized>(
        &self,
        energy: f64,
        temperature: &str,
        rng: &mut R,
    ) -> &Reaction {
        // Get temperature index for O(1) lookup
        let temp_idx = self
            .get_temp_idx(temperature)
            .expect("[sample_absorption_constituent] Temperature not found");
        let temp_reactions = self
            .reactions
            .get(temp_idx)
            .expect("[sample_absorption_constituent] No reaction data for temperature");

        // Two-pass approach WITHOUT allocation:
        // Pass 1: Compute total absorption cross section
        let mut total_xs = 0.0;
        for (&mt, reaction) in temp_reactions.iter() {
            if is_absorption_mt(mt) {
                if let Some(xs) = reaction.cross_section_at(energy) {
                    if xs > 0.0 {
                        total_xs += xs;
                    }
                }
            }
        }

        if total_xs <= 0.0 {
            panic!("sample_absorption_constituent: No absorption reactions at energy {energy} eV");
        }

        // Pass 2: Sample reaction
        let xi = rng.random_range(0.0..total_xs);
        let mut accum = 0.0;
        for (&mt, reaction) in temp_reactions.iter() {
            if is_absorption_mt(mt) {
                if let Some(xs) = reaction.cross_section_at(energy) {
                    if xs > 0.0 {
                        accum += xs;
                        if xi < accum {
                            return reaction;
                        }
                    }
                }
            }
        }

        panic!("sample_absorption_constituent: sampling failed");
    }
}
