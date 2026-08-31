//! GPU-ready cross-section container (`GpuNuclideXs`) and the
//! extraction error type (`NuclideXsError`).

use super::constants::*;

/// GPU-ready cross-section arrays extracted from a single nuclide.
#[derive(Clone, Debug, Default)]
pub struct GpuNuclideXs {
    /// `ln(energy[i])` for each grid point of the FINE (resonance) grid.
    /// Sorted ascending. Same length as `xs_elastic` / `xs_absorption` /
    /// `xs_inelastic`. For a multi-nuclide material this is the exact UNION of
    /// the per-nuclide grids (issue #88), so every isotope's resonances are
    /// preserved for the collision / nuclide-selection cross sections. For a
    /// single nuclide it is exactly that nuclide's grid (so the single-nuclide
    /// path stays bit-identical).
    pub log_energy_grid: Vec<f64>,
    /// `ln(energy[i])` for the COARSE grid that backs the per-MT inelastic
    /// buffers (`xs_inelastic_per_mt`, `yield_per_mt`). Issue #88: those are 3D
    /// `[MT_INELASTIC_COUNT × n_grid]` buffers, so putting them on the (much
    /// larger) union grid would blow GPU memory (SS316 ~2.6 GB). Inelastic is a
    /// fast-energy channel with no epithermal resonance structure, so a coarse
    /// grid (the finest single per-nuclide grid) costs no accuracy there.
    /// Equal to `log_energy_grid` for a single-nuclide material (bit-identical).
    pub coarse_log_energy_grid: Vec<f64>,
    /// Elastic-scatter macroscopic cross section in barns at each
    /// grid point. (Macroscopic if a single-nuclide pseudo-material
    /// of unit atomic density; microscopic otherwise -- caller scales.)
    pub xs_elastic: Vec<f64>,
    /// Absorption (radiative capture, MT=102) cross section. Same
    /// shape as `xs_elastic`.
    pub xs_absorption: Vec<f64>,
    /// Aggregated inelastic-scattering cross section, summed over
    /// MT 51..=91 (discrete levels + continuum). Same shape as
    /// `xs_elastic`. The kernel uses this as the probability weight
    /// for selecting the inelastic *branch* in collision sampling;
    /// once the branch is taken, `xs_inelastic_per_mt` is what
    /// drives the per-MT pick. (`xs_inelastic[i] == sum over MT
    /// slots of xs_inelastic_per_mt[slot * n_grid + i]`.)
    pub xs_inelastic: Vec<f64>,
    /// Per-MT inelastic xs, flat layout `[MT_INELASTIC_COUNT × n_grid]`.
    /// Slot `k` (offset `k * n_grid`) holds the xs for MT
    /// `MT_INELASTIC_FIRST + k` at every energy grid point. Slots for
    /// MTs the nuclide doesn't have are all-zero.
    pub xs_inelastic_per_mt: Vec<f64>,
    /// Per-MT Q-value, length `MT_INELASTIC_COUNT`. Slot `k` holds
    /// the Q-value for MT `MT_INELASTIC_FIRST + k`. Used by the
    /// kernel's level-inelastic kinematics formula
    /// `E_out = (A/(A+1))^2 * (E_in - (A+1)/A * |Q|)` once a specific
    /// MT has been sampled. Zero for MTs not present (those slots
    /// are also xs=0 so they're never sampled).
    pub q_inelastic_per_mt: Vec<f64>,
    /// Per-MT outgoing-neutron yield ν(E) tabulated on the master
    /// energy grid, flat `[MT_INELASTIC_COUNT × n_grid]`. Slot `k` at
    /// energy index `i` holds `product.product_yield.evaluate(energy_grid[i])`
    /// where `product` is the first neutron product on the MT's
    /// reaction. Slots without a neutron product carry `1.0`
    /// everywhere -- the CPU fallback default in
    /// `yamc::inelastic::sample_from_products_with_awr`. The kernel
    /// multiplies the surviving particle's weight by the interpolated
    /// value at MT-sampling time. Replaces the hardcoded `MT_YIELDS`
    /// constant lookup; for libraries with energy-dependent yields
    /// (some n,2n / n,3n evaluations) this matches CPU exactly, and
    /// for the typical constant-yield case still gives `1.0` / `2.0` /
    /// `3.0` at every grid point.
    pub yield_per_mt: Vec<f64>,
    /// Atomic weight ratio (target mass / neutron mass). Used by the
    /// elastic-scatter kinematics step in the GPU kernel.
    pub target_mass: f64,
    /// Temperature in Kelvin for the resolved evaluation. Used by
    /// the kernel's free-gas thermal scattering branch:
    /// when `E < 400 · k_B · T` (and target_mass > 1), the kernel
    /// samples a target velocity from the CXS Maxwell distribution
    /// instead of treating the target as stationary. Above the
    /// threshold the closed-form A-mass elastic kinematics is
    /// used, identical to the pre-free-gas behaviour.
    pub temperature_k: f64,
    /// Number of incident-energy points tabulated for each MT slot's
    /// angular distribution. Length `MT_INELASTIC_COUNT`. A value of
    /// `0` means no angular data was extracted for this MT (slot was
    /// missing on the nuclide, the neutron product had no
    /// `UncorrelatedAngleEnergy` distribution, or the angle table was
    /// empty); the kernel falls back to isotropic-in-CM sampling for
    /// that slot.
    pub angle_n_energies: Vec<u32>,
    /// Tight CSR (issue #104): incident-energy grids for all MT slots
    /// concatenated back to back, length `sum(angle_n_energies)`. Slot
    /// `k`'s rows start at the CSR ae-row base built downstream in
    /// `translate.rs` from the `angle_n_energies` counts.
    pub angle_energy_grid: Vec<f64>,
    /// Number of `(mu, cdf)` points per incident-energy row, one entry
    /// per ae-row (length `sum(angle_n_energies)`, parallel to
    /// `angle_energy_grid`).
    pub angle_n_mu: Vec<u32>,
    /// Tabulated `mu` (cosine of scattering angle in CM frame for
    /// `scatter_in_cm` MTs, lab frame otherwise) values, concatenated
    /// tight across all rows, length `sum(angle_n_mu)`. A row's points
    /// start at the per-row mu-point base built downstream from
    /// `angle_n_mu`.
    pub angle_mu: Vec<f64>,
    /// Cumulative distribution function values, same shape as
    /// `angle_mu`. Used directly for inversion sampling: draw `xi ∈
    /// [0, 1]`, find the bracket where `cdf[j] <= xi <= cdf[j + 1]`,
    /// then interpolate within that bracket.
    pub angle_cdf: Vec<f64>,
    /// Per-point PDF values matching `angle_mu` / `angle_cdf`, same
    /// shape. Used by the kernel's quadratic LinLin CDF inversion.
    pub angle_pdf: Vec<f64>,
    /// Per-ae-row interpolation discriminant, one entry per ae-row
    /// (length `sum(angle_n_energies)`). Values are
    /// `ANGLE_INTERP_HISTOGRAM` or `ANGLE_INTERP_LINLIN`.
    pub angle_interp: Vec<u32>,
    /// Per-MT outgoing-energy distribution kind, length
    /// `MT_INELASTIC_COUNT`. Values are `EOUT_KIND_LEVEL_INELASTIC`
    /// (closed-form Q-value formula, the original GPU path) or
    /// `EOUT_KIND_CONTINUOUS_TABULAR` (kernel samples from
    /// `eout_x` / `eout_cdf`). Slots without continuum data default
    /// to `EOUT_KIND_LEVEL_INELASTIC` so the kernel falls back to
    /// the closed form.
    pub eout_kind: Vec<u32>,
    /// Number of incident-energy points tabulated for each MT slot's
    /// outgoing-energy distribution. Length `MT_INELASTIC_COUNT`. A
    /// value of `0` means no continuum data was extracted; the
    /// kernel falls back to the closed-form energy regardless of
    /// `eout_kind`.
    pub eout_n_energies: Vec<u32>,
    /// Per-MT `histogram_interp` flag from the outer
    /// `EnergyDistribution::ContinuousTabular`. `1` makes the kernel
    /// suppress the stochastic E_in bracket pick and the bracket-bound
    /// stretch, mirroring CPU's
    /// `ContinuousTabular::sample`. Length `MT_INELASTIC_COUNT`.
    pub eout_histogram_interp: Vec<u32>,
    /// Tight CSR (issue #104): incident-energy grids for all MT slots'
    /// outgoing-energy distributions concatenated back to back, length
    /// `sum(eout_n_energies)`. Same CSR conventions as
    /// `angle_energy_grid` (per-slot ae-row bases built downstream).
    pub eout_energy_grid: Vec<f64>,
    /// Number of `(E_out, cdf)` points per incident-energy row, one
    /// entry per ae-row (length `sum(eout_n_energies)`).
    pub eout_n_x: Vec<u32>,
    /// Tabulated outgoing-energy values, concatenated tight across all
    /// rows, length `sum(eout_n_x)`. Frame semantics follow
    /// `scatter_in_cm`: CM-frame `E_out` if the flag is set, lab-frame
    /// otherwise.
    pub eout_x: Vec<f64>,
    /// Per-point PDF values for the outgoing energy, same shape as
    /// `eout_x`. The kernel uses this for the quadratic LinLin CDF
    /// inversion that mirrors CPU's `Tabular::sample`. PDF values are
    /// pre-scaled by `1 / cdf[last]` so the renormalised `(p, cdf)`
    /// pair stays consistent (`∫ p dx = c[last] = 1`).
    pub eout_p: Vec<f64>,
    /// Cumulative distribution function values for the outgoing
    /// energy, same shape as `eout_x`. CDF-inversion bracket search;
    /// post-bracket the kernel dispatches on `eout_interp` for the
    /// `c → x` formula (Histogram or LinLin quadratic).
    pub eout_cdf: Vec<f64>,
    /// Per-(MT slot, E_in slice) interpolation discriminant. `0` =
    /// Histogram (kernel uses `x_k + (r − c_k) / p_k`), `1` = LinLin
    /// (kernel uses the quadratic LinLin formula). Same encoding as
    /// `angle_interp` / `corr_mu_interp`. One entry per ae-row
    /// (length `sum(eout_n_energies)`).
    pub eout_interp: Vec<u32>,
    /// Per-(MT slot, E_in slice) discrete-line prefix count. The
    /// first `n_discrete` bins of each Tabular are discrete photon
    /// lines (delta functions). The kernel detects when the sampled
    /// bin is `< n_discrete` and returns the bin endpoint instead of
    /// the CDF-inverted position, also skipping the bracket-bound
    /// stretch. One entry per ae-row (length `sum(eout_n_energies)`).
    pub eout_n_discrete: Vec<u32>,
    /// Per-MT correlated-distribution incident-energy count (TOTAL across
    /// components, i.e. `n_components * n_per_comp`), length
    /// `MT_INELASTIC_COUNT`. Non-zero only when `eout_kind ==
    /// EOUT_KIND_CORRELATED` for that slot.
    pub corr_n_energies: Vec<u32>,
    /// Per-MT count of equally-weighted correlated components (issue #111),
    /// length `MT_INELASTIC_COUNT`. `>= 2` when the neutron product carries
    /// several applicability-gated `CorrelatedAngleEnergy` laws (F19 MT16
    /// n,2n); the kernel/twin draw one uniform per collision to pick a
    /// component, and component `c` occupies the `corr_n_energies /
    /// corr_n_components` rows at `corr_ae_offset[slot] + c * n_per_comp`.
    pub corr_n_components: Vec<u32>,
    /// Tight CSR (issue #104): correlated incident-energy grids for all
    /// MT slots concatenated, length `sum(corr_n_energies)`.
    pub corr_energy_grid: Vec<f64>,
    /// Correlated outgoing-energy point count per incident-energy row,
    /// one entry per ae-row (length `sum(corr_n_energies)`).
    pub corr_n_x: Vec<u32>,
    /// Correlated outgoing-energy values, concatenated tight across all
    /// rows, length `sum(corr_n_x)`.
    pub corr_x: Vec<f64>,
    /// Per-MT correlated outgoing-energy CDF, same shape as
    /// `corr_x`. Used like `eout_cdf` -- the kernel walks the CDF
    /// to invert, then linearly interpolates in `c → x`.
    pub corr_cdf: Vec<f64>,
    /// Per-point PDF for `corr_x` / `corr_cdf`, same shape as `corr_x`.
    /// Needed by the
    /// kernel's quadratic LinLin CDF inversion that mirrors CPU's
    /// `Tabular::sample_with_discrete_info`. PDF values are pre-scaled
    /// by `1 / cdf[last]` so the renormalised `(p, cdf)` pair stays
    /// consistent (`∫ p dx = c[last] = 1`).
    pub corr_p: Vec<f64>,
    /// Per (MT slot, E_in slice) interpolation discriminant. `0` =
    /// Histogram (kernel uses `x_k + (r − c_k) / p_k`), `1` = LinLin
    /// (quadratic LinLin formula). Same encoding as `angle_interp` /
    /// `eout_interp`. One entry per ae-row (length
    /// `sum(corr_n_energies)`).
    pub corr_interp: Vec<u32>,
    /// Per-ae-row discrete-line prefix count. The first `n_discrete`
    /// bins are discrete delta functions; the kernel returns the bin
    /// endpoint (no CDF interpolation) for those. One entry per ae-row
    /// (length `sum(corr_n_energies)`).
    pub corr_n_discrete: Vec<u32>,
    /// Correlated mu count per `(E_in, E_out)` bin, one entry per
    /// outgoing-energy point (length `sum(corr_n_x)`). Zero for bins
    /// without an angular sub-table -- the kernel falls back to
    /// isotropic for those.
    pub corr_n_mu: Vec<u32>,
    /// Correlated mu values, concatenated tight across all `(E_in,
    /// E_out)` bins, length `sum(corr_n_mu)`. Frame semantics follow
    /// `scatter_in_cm` (CM if the flag is set, lab otherwise).
    pub corr_mu: Vec<f64>,
    /// Per-MT correlated mu CDF, same shape as `corr_mu`. The kernel
    /// inverts the CDF and then dispatches on `corr_mu_interp` for the
    /// `(c → x)` formula: histogram brackets use linear-in-c,
    /// LinLin brackets use the quadratic LinLin formula, matching
    /// CPU's `Tabular::sample`.
    pub corr_mu_cdf: Vec<f64>,
    /// Per-point PDF for `corr_mu` / `corr_mu_cdf`, same shape as
    /// `corr_mu`. Used by the kernel's quadratic LinLin CDF inversion.
    /// PDF values are normalised by the sub-table integral (so that
    /// `sum(p · Δx) ≈ 1`).
    pub corr_mu_pdf: Vec<f64>,
    /// Per-`(E_in, E_out)`-bin interpolation discriminant, one entry
    /// per outgoing-energy point (length `sum(corr_n_x)`). Values are
    /// `ANGLE_INTERP_HISTOGRAM` or `ANGLE_INTERP_LINLIN`, matching
    /// `yamc_nuclide::secondary_correlated::Interpolation`.
    pub corr_mu_interp: Vec<u32>,
    /// Per-MT flag indicating whether the tabulated angular
    /// distribution is in the centre-of-mass frame (`1`) or the lab
    /// frame (`0`). Length `MT_INELASTIC_COUNT`. Mirrors
    /// `Reaction::scatter_in_cm`. When set, the kernel applies the
    /// two-body CM→lab conversion to the sampled `(E, mu)` pair using
    /// the closed-form level-inelastic CM-frame outgoing energy.
    pub scatter_in_cm: Vec<u32>,
    /// Total fission macroscopic cross section, summed across every
    /// fission MT (18 / 19 / 20 / 21 / 38) the nuclide carries. Same
    /// shape as `xs_elastic` (`n_grid` entries). Used by the kernel
    /// to pick the fission *branch* in collision sampling
    /// (4-way: elastic / inelastic / fission / absorption). When the
    /// branch is taken, the kernel multiplies the surviving particle's
    /// weight by `nu_bar` (variance-reduction equivalent of emitting
    /// `nu_bar` independent prompt neutrons) and resamples the
    /// outgoing energy from a Watt spectrum.
    pub xs_fission: Vec<f64>,
    /// Average prompt neutrons per fission, ν̄(E), evaluated at every
    /// master-grid energy point. Same shape as `xs_elastic`. Used as
    /// the per-step weight multiplier when the kernel samples a
    /// fission collision.
    pub nu_bar: Vec<f64>,
    /// Delayed-neutron fraction `beta(E) = nu_d(E) / nu_t(E)`, evaluated at every
    /// master-grid energy point (issue #364). Same shape as `nu_bar`. Per fission
    /// progeny the kernel draws one uniform against this and takes the outgoing
    /// energy from `fission_eout_delayed_*` instead of the prompt table when it
    /// lands below. Zero everywhere for a material whose nuclides carry no delayed
    /// data, which costs no draw.
    pub beta_delayed: Vec<f64>,
    /// Watt-spectrum `a` parameter (eV) used to sample the prompt
    /// fission neutron's outgoing energy. The kernel uses the same
    /// rejection algorithm as `yamc_nuclide::sampling::sample_watt_spectrum_params`.
    /// MVP: a single representative parameter per material --
    /// extracted from the dominant fission MT's neutron product if
    /// present, otherwise defaults to 0.988 MeV (typical thermal-
    /// fission Watt parameter -- close enough across actinides for a
    /// first cut).
    pub fission_watt_a: f64,
    /// Watt-spectrum `b` parameter (1/eV). MVP default 2.249e-6 1/eV
    /// (typical thermal Watt).
    pub fission_watt_b: f64,
    /// Discriminant for the fission outgoing-energy sampler. Values:
    /// - `EOUT_KIND_CONTINUOUS_TABULAR` (1): kernel samples from
    ///   `fission_eout_x` / `fission_eout_cdf` (also covers the
    ///   `CorrelatedAngleEnergy` E_out marginal -- same buffer encoding).
    /// - `EOUT_KIND_MAXWELL` (6): kernel samples `√E·exp(-E/θ)` via the
    ///   shared Maxwell rejection helper. Tight CSR (issue #104): each E_in
    ///   row carries one point (`fission_eout_n_x[i] == 1`); the tabulated
    ///   `θ(E_in)` lives in that single `fission_eout_x` slot per row and the
    ///   scalar restriction energy `u` in the material's row-0 `fission_eout_cdf`
    ///   slot.
    /// - `EOUT_KIND_EVAPORATION` (4): kernel samples `E·exp(-E/θ)` via the
    ///   shared Evaporation rejection helper. Same θ / `u` packing as Maxwell.
    /// - `EOUT_KIND_WATT` (7): kernel uses the Watt-rejection branch
    ///   with `fission_watt_a` / `fission_watt_b`. Retained as the
    ///   fallback for nuclides whose χ uses none of the above encodings.
    pub fission_eout_kind: u32,
    /// Number of populated incident-energy points in the fission
    /// outgoing-energy table. Zero for Watt-fallback / non-fissionable
    /// materials.
    pub fission_eout_n_energies: u32,
    /// Incident-energy grid for the fission outgoing-energy table.
    /// Tight (issue #104): length `fission_eout_n_energies` (no padding).
    pub fission_eout_energy_grid: Vec<f64>,
    /// Per (E_in_idx) outgoing-energy point count, length
    /// `fission_eout_n_energies`.
    pub fission_eout_n_x: Vec<u32>,
    /// Tabulated outgoing-energy values, tight CSR (issue #104): rows
    /// concatenated back-to-back, row `i` occupying `n_x[i]` points.
    /// Length `sum(fission_eout_n_x)`. The per-row CSR base is built at
    /// translate time (`fission_eout_x_offset`).
    pub fission_eout_x: Vec<f64>,
    /// Cumulative distribution for `fission_eout_x`, same shape.
    pub fission_eout_cdf: Vec<f64>,
    /// Probability density for `fission_eout_x`, same shape, normalized
    /// by the same `cdf_max` as `fission_eout_cdf`. Zero-filled for rows
    /// without a usable PDF (the sampler then falls back to linear-in-c).
    pub fission_eout_p: Vec<f64>,
    /// Per-incident-energy-row interpolation discriminant for the fission
    /// E_out inversion, one entry per ae-row (length
    /// `fission_eout_n_energies`). `0` = histogram, `1` = lin-lin
    /// (matching `TabulatedInterp`).
    pub fission_eout_interp: Vec<u32>,
    /// The DELAYED fission spectrum, same six-buffer encoding as the prompt
    /// `fission_eout_*` fields above (issue #364). It is the yield-weighted fold of
    /// the evaluation's six delayed groups, so it is always
    /// `EOUT_KIND_CONTINUOUS_TABULAR` when present and all-zero / empty when the
    /// material has no delayed data. Translate time appends these as a SECOND chi
    /// row per material, so the kernel reads material `m`'s prompt spectrum at row
    /// `2m` and its delayed spectrum at row `2m + 1`.
    pub fission_eout_delayed_kind: u32,
    pub fission_eout_delayed_n_energies: u32,
    pub fission_eout_delayed_energy_grid: Vec<f64>,
    pub fission_eout_delayed_n_x: Vec<u32>,
    pub fission_eout_delayed_x: Vec<f64>,
    pub fission_eout_delayed_cdf: Vec<f64>,
    pub fission_eout_delayed_p: Vec<f64>,
    pub fission_eout_delayed_interp: Vec<u32>,
    /// Kalbach-Mann incident-energy point count per MT slot, length
    /// `MT_INELASTIC_COUNT`. Non-zero only when `eout_kind ==
    /// EOUT_KIND_KALBACH_MANN` for that slot. The per-slot KM table
    /// is structured the same way as `corr_*`: per (E_in, E_out)
    /// PDF/CDF for the energy axis plus per-(E_in, E_out) (r, a)
    /// Kalbach-Mann parameters for the mu sampler.
    pub km_n_energies: Vec<u32>,
    /// Tight CSR (issue #104): Kalbach-Mann incident-energy grids for
    /// all MT slots concatenated, length `sum(km_n_energies)`.
    pub km_energy_grid: Vec<f64>,
    /// Per-(E_in) interpolation discriminant for the E_out axis, one
    /// entry per ae-row (length `sum(km_n_energies)`). `0` = histogram,
    /// `1` = lin-lin (matching `secondary_kalbach::Interpolation`).
    pub km_interp: Vec<u32>,
    /// Per-(E_in) discrete-pre-continuous count, one entry per ae-row
    /// (length `sum(km_n_energies)`). Most evaluations use 0; non-zero
    /// values cap a discrete-photon-style prefix on the E_out grid that
    /// the kernel walks before the continuous portion (mirrors CPU's
    /// `n_discrete`).
    pub km_n_discrete: Vec<u32>,
    /// Per-(E_in) E_out-point count, one entry per ae-row (length
    /// `sum(km_n_energies)`).
    pub km_n_x: Vec<u32>,
    /// Tabulated outgoing-energy values, concatenated tight across all
    /// rows, length `sum(km_n_x)`. CM-frame semantics follow
    /// `scatter_in_cm` for the slot.
    pub km_x: Vec<f64>,
    /// Per-(E_in, E_out) PDF, same shape as `km_x`. Used by the
    /// kernel's CDF inversion when `interp == LinLin` (quadratic
    /// formula) and the histogram path (`e_out = e_l_k + (r1 -
    /// c_k) / p_l_k`).
    pub km_p: Vec<f64>,
    /// Per-(E_in, E_out) CDF, same shape as `km_x`.
    pub km_c: Vec<f64>,
    /// Per-(E_in, E_out) Kalbach-Mann r parameter (compound vs
    /// precompound mixture coefficient), same shape as `km_x`.
    pub km_r: Vec<f64>,
    /// Per-(E_in, E_out) Kalbach-Mann a parameter (slope), same
    /// shape as `km_x`.
    pub km_a: Vec<f64>,
    /// Per-MT Evaporation parameter count, length `MT_INELASTIC_COUNT`.
    /// Non-zero only when `eout_kind == EOUT_KIND_EVAPORATION` for
    /// that slot. The kernel samples
    /// `p(E_out) ~ E_out · exp(-E_out/θ(E_in))` for
    /// `0 < E_out < E_in - u` via the standard Maxwell-style
    /// rejection algorithm, with `θ(E_in)` linearly interpolated on
    /// the slot's tabulated grid.
    pub evap_n_energies: Vec<u32>,
    /// Per-MT Evaporation component count, length `MT_INELASTIC_COUNT`.
    /// `>= 2` when the neutron product carries several equally-weighted
    /// Evaporation laws (the (n,xn) channels of a handful of endf-b8.1
    /// nuclides); the kernel then draws one uniform per collision to pick a
    /// component, mirroring the CPU's per-collision applicability sampling.
    /// `0`/`1` mean a single curve (no selector draw).
    pub evap_n_components: Vec<u32>,
    /// Tabulated incident-energy grid for the evaporation θ(E_in)
    /// parameter. Tight variable-length layout (issue #104): exactly
    /// `sum(evap_n_energies)` rows back to back, no per-axis padding.
    /// The per-(slab,MT) CSR base (`evap_ae_offset`) is built at
    /// concatenation time in `translate.rs`.
    pub evap_energy_grid: Vec<f64>,
    /// Component-major tabulated θ values. Tight variable-length layout
    /// (issue #104): exactly `sum(evap_n_components * evap_n_energies)` rows.
    /// Component `c`'s row begins at `evap_theta_offset[slot] + c *
    /// evap_n_energies[slot]`, where `evap_theta_offset` is built in
    /// `translate.rs`. `MAX_EVAP_COMPONENTS` bounds the kernel's component
    /// loop; it is not a buffer stride.
    pub evap_theta: Vec<f64>,
    /// Per-MT restriction energy `u` (eV), length `MT_INELASTIC_COUNT`.
    /// `E_out` is constrained to `[0, E_in - u]`.
    pub evap_u: Vec<f64>,
    /// Per-MT NBodyPhaseSpace particle count (3, 4, or 5), length
    /// `MT_INELASTIC_COUNT`. `0` means the slot isn't NBPS-encoded
    /// (the kernel falls through to the closed-form path).
    pub nbps_n_bodies: Vec<u32>,
    /// Per-MT NBodyPhaseSpace total mass ratio (sum of AWRs of all
    /// products), length `MT_INELASTIC_COUNT`. Only meaningful when
    /// `nbps_n_bodies > 0`.
    pub nbps_total_mass: Vec<f64>,
    /// Per-MT Maxwell parameter count, length `MT_INELASTIC_COUNT`.
    /// Non-zero only when `eout_kind == EOUT_KIND_MAXWELL` for that
    /// slot. The kernel samples
    /// `p(E_out) ~ sqrt(E_out) · exp(-E_out / θ(E_in))` for
    /// `0 < E_out < E_in - u` via the standard 3-uniform Maxwell
    /// rejection (`E_out = -θ · (ln r1 + ln r2 · cos²(π/2 · r3))`,
    /// retry until `E_out ≤ E_in - u`). `θ(E_in)` is linearly
    /// interpolated on the slot's tabulated grid.
    pub maxwell_n_energies: Vec<u32>,
    /// Tabulated incident-energy grid for the Maxwell θ(E_in)
    /// parameter. Tight variable-length layout (issue #104): exactly
    /// `sum(maxwell_n_energies)` rows back to back, no per-axis
    /// padding. The per-(slab,MT) CSR base (`maxwell_ae_offset`) is built
    /// at concatenation time in `translate.rs`.
    pub maxwell_energy_grid: Vec<f64>,
    /// Tabulated θ values for the Maxwell distribution at each
    /// incident-energy index, same shape as `maxwell_energy_grid`.
    pub maxwell_theta: Vec<f64>,
    /// Per-MT restriction energy `u` (eV) for Maxwell, length
    /// `MT_INELASTIC_COUNT`. `E_out` is constrained to
    /// `[0, E_in - u]`.
    pub maxwell_u: Vec<f64>,
    /// Per-MT Watt parameter count, length `MT_INELASTIC_COUNT`.
    /// Non-zero only when `eout_kind == EOUT_KIND_WATT` for that
    /// slot. The kernel samples
    /// `p(E_out) ~ exp(-E_out / a(E_in)) · sinh(sqrt(b(E_in) · E_out))`
    /// for `0 < E_out < E_in - u` via the Maxwell-product trick
    /// (`w = -a · (ln r1 + ln r2 · cos²(π/2 · r3))`, then
    /// `E = w + a²b/4 + (2 r4 - 1) · sqrt(a²b · w)`, retry until
    /// `E ≤ E_in - u`). Both `a(E_in)` and `b(E_in)` are linearly
    /// interpolated on the slot's tabulated grid.
    pub watt_n_energies: Vec<u32>,
    /// Tabulated incident-energy grid for the Watt `a(E_in)` and
    /// `b(E_in)` parameters. Tight variable-length layout (issue #104):
    /// exactly `sum(watt_n_energies)` rows back to back, no per-axis
    /// padding. `a` and `b` share this grid; the per-(slab,MT) CSR base
    /// (`watt_ae_offset`) is built at concatenation time in `translate.rs`.
    pub watt_energy_grid: Vec<f64>,
    /// Tabulated `a` values (eV) for the Watt distribution at each
    /// incident-energy index, same shape as `watt_energy_grid`.
    pub watt_a: Vec<f64>,
    /// Tabulated `b` values (1/eV) for the Watt distribution at each
    /// incident-energy index, same shape as `watt_energy_grid`.
    pub watt_b: Vec<f64>,
    /// Per-MT restriction energy `u` (eV) for Watt, length
    /// `MT_INELASTIC_COUNT`. `E_out` is constrained to
    /// `[0, E_in - u]`.
    pub watt_u: Vec<f64>,
    /// Per-(material, nuclide) URR (unresolved resonance region) flags,
    /// packed `[n_nuclides × URR_META_COLS]` u32, one row per nuclide of the
    /// material in `weighted` order (issue #210: URR is applied to EVERY
    /// in-range URR nuclide, so this is keyed on the slab, not one dominant
    /// nuclide). `URR_META_PRESENT = 0` rows are non-URR nuclides (the kernel
    /// skips them and keeps their smooth XS). See `URR_META_*` for the
    /// columns; `URR_META_ZA` carries the per-nuclide stream key.
    pub urr_meta: Vec<u32>,
    /// Tight CSR (issue #104): per-slab URR energy grids (eV) concatenated
    /// back to back, length `sum(urr_meta[.., URR_META_N_ENERGIES])`. The
    /// per-slab base is built downstream in `translate.rs`.
    pub urr_energy_grid: Vec<f64>,
    /// Per-slab URR cumulative-distribution-function values, concatenated
    /// tight (`[n_energies × n_cdf]` per URR slab). Band `j` covers
    /// `r ∈ [cdf[..j-1], cdf[..j])` (where `cdf[..-1] = 0` for band 0).
    pub urr_cdf: Vec<f64>,
    /// Per-slab URR table cross-sections, concatenated tight
    /// (`[n_energies × n_cdf × URR_XS_COLS]` per URR slab). `URR_XS_*`
    /// constants select total / elastic / fission / n_gamma. Values are
    /// absolute XS when `urr_meta[.., MULTIPLY_SMOOTH] == 0`, factors when
    /// `== 1`.
    pub urr_xs: Vec<f64>,
    /// Per-nuclide atom density (atoms/barn-cm), length `n_nuclides`, one
    /// entry per slab in the same order as `urr_meta`'s rows. Used to scale a
    /// URR nuclide's perturbed micro XS up to a macroscopic contribution.
    /// Zero for non-URR nuclides. The smooth baselines the URR delta is taken
    /// against come from `nuc_partial_xs` (already density-weighted
    /// macroscopic), so no per-nuclide smooth micro buffer is needed (#210).
    pub urr_atom_density: Vec<f64>,
}

impl GpuNuclideXs {
    /// Build a synthetic all-zero "void material" slot for a
    /// material-less (void) cell on the GPU.
    ///
    /// Every cross-section row is zero, so the kernel's
    /// `sigma_t = sigma_e + sigma_a + sigma_i + sigma_f` evaluates to
    /// `0` for a cell mapped to this slot. The free-flight sample then
    /// streams the particle to the next surface crossing with no
    /// collision (`d_xs = -ln(xi) / 0 = +inf > d_boundary`), and the
    /// collision block -- which is the only place a per-material XS or
    /// distribution buffer is divided by `sigma_t` or sampled -- is
    /// gated behind `collide_first`, so it never runs in void. Track-
    /// length flux still accrues (the per-step scoring uses the
    /// surface-crossing distance with `score = 1.0`); reaction-rate /
    /// total / heating tallies score `0` (their `score_xs` reads these
    /// zero rows). This matches the CPU's void (delta-tracking-free)
    /// streaming semantics exactly.
    ///
    /// Per-slot scalar / count arrays are sized to the same
    /// `[MT_INELASTIC_COUNT]` per-material stride a real extracted
    /// material uses (see `extract_material_xs`); the tight variable-
    /// length distribution arrays (angle / eout / corr / km / evap /
    /// maxwell / watt / fission / URR, issue #104) carry no rows, so
    /// they are empty (the void slot contributes zero ae-rows / points
    /// to the concatenated CSR buffers). Either way the host launcher's
    /// per-material length assertions pass when this slot is appended.
    /// `urr_meta` is left all-zero (`URR_META_PRESENT = 0`) so the
    /// kernel's URR branch is skipped for the void slot. `target_mass`
    /// is a benign `1.0` (never read -- collision kinematics don't run
    /// in void).
    pub fn void(log_energy_grid: Vec<f64>, coarse_log_energy_grid: Vec<f64>) -> Self {
        let n_grid = log_energy_grid.len();
        let n_coarse = coarse_log_energy_grid.len();
        let mt = MT_INELASTIC_COUNT;
        Self {
            // Void slot is all-zero. The per-MT inelastic buffers ride the
            // COARSE grid (#88), so the void slab's length matches the
            // per-nuclide pool's coarse stride the kernel indexes with; the
            // per-material collision buffers and URR stay on the fine grid.
            coarse_log_energy_grid,
            log_energy_grid,
            xs_elastic: vec![0.0; n_grid],
            xs_absorption: vec![0.0; n_grid],
            xs_inelastic: vec![0.0; n_grid],
            xs_inelastic_per_mt: vec![0.0; mt * n_coarse],
            q_inelastic_per_mt: vec![0.0; mt],
            // `yield_per_mt` defaults to 1.0 in a real extraction, but
            // it is only read inside the inelastic collision branch
            // (never reached in void), so all-zero is equally safe and
            // keeps the void slot uniformly zero.
            yield_per_mt: vec![0.0; mt * n_coarse],
            target_mass: 1.0,
            temperature_k: 0.0,
            // Tight CSR (issue #104): the void slot carries no inelastic
            // angular rows, so the per-slot counts are zero and the per-row /
            // per-point arrays are empty.
            angle_n_energies: vec![0u32; mt],
            angle_energy_grid: Vec::new(),
            angle_n_mu: Vec::new(),
            angle_mu: Vec::new(),
            angle_cdf: Vec::new(),
            angle_pdf: Vec::new(),
            angle_interp: Vec::new(),
            eout_kind: vec![0u32; mt],
            eout_n_energies: vec![0u32; mt],
            eout_histogram_interp: vec![0u32; mt],
            // Tight CSR (issue #104): no outgoing-energy rows on the void slot.
            eout_energy_grid: Vec::new(),
            eout_n_x: Vec::new(),
            eout_x: Vec::new(),
            eout_p: Vec::new(),
            eout_cdf: Vec::new(),
            eout_interp: Vec::new(),
            eout_n_discrete: Vec::new(),
            corr_n_energies: vec![0u32; mt],
            corr_n_components: vec![0u32; mt],
            // Tight CSR (issue #104): no correlated rows on the void slot.
            corr_energy_grid: Vec::new(),
            corr_n_x: Vec::new(),
            corr_x: Vec::new(),
            corr_cdf: Vec::new(),
            corr_p: Vec::new(),
            corr_interp: Vec::new(),
            corr_n_discrete: Vec::new(),
            corr_n_mu: Vec::new(),
            corr_mu: Vec::new(),
            corr_mu_cdf: Vec::new(),
            corr_mu_pdf: Vec::new(),
            corr_mu_interp: Vec::new(),
            scatter_in_cm: vec![0u32; mt],
            xs_fission: vec![0.0; n_grid],
            nu_bar: vec![0.0; n_grid],
            beta_delayed: vec![0.0; n_grid],
            fission_watt_a: 0.0,
            fission_watt_b: 0.0,
            fission_eout_kind: 0,
            fission_eout_n_energies: 0,
            // Tight CSR (issue #104): no fission distribution -> zero rows,
            // zero points, so every buffer is empty.
            fission_eout_energy_grid: Vec::new(),
            fission_eout_n_x: Vec::new(),
            fission_eout_x: Vec::new(),
            fission_eout_cdf: Vec::new(),
            fission_eout_p: Vec::new(),
            fission_eout_interp: Vec::new(),
            fission_eout_delayed_kind: 0,
            fission_eout_delayed_n_energies: 0,
            fission_eout_delayed_energy_grid: Vec::new(),
            fission_eout_delayed_n_x: Vec::new(),
            fission_eout_delayed_x: Vec::new(),
            fission_eout_delayed_cdf: Vec::new(),
            fission_eout_delayed_p: Vec::new(),
            fission_eout_delayed_interp: Vec::new(),
            km_n_energies: vec![0u32; mt],
            // Tight CSR (issue #104): no Kalbach-Mann rows on the void slot.
            km_energy_grid: Vec::new(),
            km_interp: Vec::new(),
            km_n_discrete: Vec::new(),
            km_n_x: Vec::new(),
            km_x: Vec::new(),
            km_p: Vec::new(),
            km_c: Vec::new(),
            km_r: Vec::new(),
            km_a: Vec::new(),
            evap_n_energies: vec![0u32; mt],
            evap_n_components: vec![0u32; mt],
            // Tight layout (issue #104): the void slot carries no Evaporation
            // rows, so the per-E_in / component-major arrays are empty.
            evap_energy_grid: Vec::new(),
            evap_theta: Vec::new(),
            evap_u: Vec::new(),
            nbps_n_bodies: vec![0u32; mt],
            nbps_total_mass: vec![0.0; mt],
            maxwell_n_energies: vec![0u32; mt],
            // Tight layout (issue #104): the void slot carries no Maxwell rows.
            maxwell_energy_grid: Vec::new(),
            maxwell_theta: Vec::new(),
            maxwell_u: vec![0.0; mt],
            watt_n_energies: vec![0u32; mt],
            // Tight layout (issue #104): the void slot carries no Watt rows.
            watt_energy_grid: Vec::new(),
            watt_a: Vec::new(),
            watt_b: Vec::new(),
            watt_u: vec![0.0; mt],
            // One URR slab for the void's single (pseudo-)nuclide, PRESENT = 0
            // (issue #210: URR is slab-keyed, and the void slab is appended
            // last so it aligns 1:1 with the nuclide-selection void slab).
            urr_meta: vec![0u32; URR_META_COLS],
            // Tight CSR layout (issue #104): the void slot carries no URR
            // energy points, so the energy / cdf / xs arrays are empty
            // (n_energies = 0 in `urr_meta`).
            urr_energy_grid: Vec::new(),
            urr_cdf: Vec::new(),
            urr_xs: Vec::new(),
            urr_atom_density: vec![0.0],
        }
    }
}

/// Errors from cross-section extraction.
#[derive(Debug)]
pub enum NuclideXsError {
    /// The nuclide doesn't have data loaded for the requested temperature.
    TemperatureNotLoaded(String),
    /// The nuclide is missing the parent energy grid for this temperature.
    MissingEnergyGrid(String),
    /// One of the requested MT reactions isn't present.
    MissingReaction { mt: i32, temperature: String },
    /// The atomic-weight ratio isn't set on the nuclide.
    MissingAtomicWeightRatio,
    /// `extract_material_xs` was called with no nuclides.
    EmptyMaterial,
    /// Atomic densities sum to zero -- material has no mass.
    ZeroTotalDensity,
    /// The nuclide carries neutron-emitting scattering MTs that have no slot in
    /// [`MT_SLOTS`](crate::neutron::xs::MT_SLOTS), and they are big enough to
    /// matter (issue #106).
    ///
    /// The kernel can only sample MTs it has a slot for. An unslotted channel's
    /// cross section is not in `xs_inelastic`, so it lands in the derived
    /// absorption and the GPU kills neutrons the CPU would scatter. Refusing is
    /// the honest outcome: `compute='auto'` falls back to the CPU, which
    /// samples every channel.
    UnslottedScatterMts {
        nuclide: String,
        mts: Vec<i32>,
        max_fraction: f64,
    },
}

impl std::fmt::Display for NuclideXsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TemperatureNotLoaded(t) => {
                write!(f, "temperature {t} not loaded for this nuclide")
            }
            Self::MissingEnergyGrid(t) => {
                write!(f, "no energy grid for temperature {t}")
            }
            Self::MissingReaction { mt, temperature } => {
                write!(f, "MT={mt} not present at temperature {temperature}")
            }
            Self::MissingAtomicWeightRatio => {
                write!(f, "nuclide has no atomic_weight_ratio")
            }
            Self::EmptyMaterial => {
                write!(f, "material aggregation called with no nuclides")
            }
            Self::ZeroTotalDensity => {
                write!(f, "material atomic densities sum to zero")
            }
            Self::UnslottedScatterMts {
                nuclide,
                mts,
                max_fraction,
            } => {
                let list = mts
                    .iter()
                    .map(|mt| mt.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "nuclide `{nuclide}` carries neutron-emitting MT {list}, which the GPU \
                     kernel has no slot for and would count as absorption \
                     ({:.2}% of this nuclide's total cross section at its worst energy). \
                     Run on the CPU, which samples every channel.",
                    max_fraction * 100.0
                )
            }
        }
    }
}

impl std::error::Error for NuclideXsError {}
