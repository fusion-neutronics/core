# Arrow file format

Each nuclide or element is stored as a directory of Arrow IPC files.  Every
`.arrow` file is a self-contained table that can be independently
memory-mapped.  The format is simulation-ready: synthesized MTs, FastXSGrid
lookup tables, and log-space cross sections are pre-computed.

## Neutron data: `{name}.arrow/`

```text
Li6.arrow/
├── version.json
├── nuclide.arrow
├── reactions.arrow
├── products.arrow
├── distributions.arrow
├── fast_xs.arrow
├── urr.arrow            (optional)
├── total_nu.arrow       (optional)
├── fission_photon.arrow (optional)
└── covariance.arrow     (optional)
```

### version.json

Format metadata written at conversion time.

```json
{
  "format_version": 1,
  "library": "fendl-3.2c",
  "converter_version": "0.1.0",
  "created_utc": "2026-02-08T14:30:00Z"
}
```

The reader checks `format_version` and rejects incompatible versions.

### nuclide.arrow

One row of nuclide-level metadata and energy grids.

| Column | Type | Description |
|---|---|---|
| `name` | `utf8` | Nuclide name in GNDS format (e.g. `"Li6"`) |
| `Z` | `int32` | Atomic number |
| `A` | `int32` | Mass number |
| `atomic_weight_ratio` | `float64` | Atomic weight ratio |
| `temperatures` | `list<utf8>` | Temperature labels (`["294K", "600K", ...]`) |
| `kTs` | `list<float64>` | kT values in eV |
| `energy_temperatures` | `list<utf8>` | Temperature keys for energy grids |
| `energy_values` | `list<list<float64>>` | Energy grids per temperature |

File-level metadata: `filetype=data_neutron`, `version=4.0`

### reactions.arrow

One row per reaction, including synthesized MTs (1, 3, 4, 27, 101).

| Column | Type | Description |
|---|---|---|
| `mt` | `int32` | ENDF MT reaction number |
| `label` | `utf8` | Reaction name |
| `Q_value` | `float64` | Q-value in eV |
| `center_of_mass` | `bool` | Center-of-mass frame flag |
| `redundant` | `bool` | Redundant reaction flag (True for synthesized MTs) |
| `xs_temperatures` | `list<utf8>` | Temperature keys |
| `xs_values` | `list<list<float64>>` | Cross-section arrays per temperature |
| `xs_threshold_idx` | `list<int32>` | Threshold index per temperature |

### products.arrow

One row per product across all reactions.

| Column | Type | Description |
|---|---|---|
| `reaction_mt` | `int32` | Parent reaction MT |
| `product_idx` | `int32` | Product index within the reaction |
| `particle` | `utf8` | Particle type (`"neutron"`, `"photon"`, etc.) |
| `emission_mode` | `utf8` | `"prompt"`, `"delayed"`, or `"total"` |
| `decay_rate` | `float64` | Decay rate (0.0 if prompt) |
| `n_distribution` | `int32` | Number of angle-energy distributions |
| `yield_type` | `utf8` | `"Tabulated1D"` or `"Polynomial"` |
| `yield_data` | `list<float64>` | Yield values (flat) |
| `yield_shape` | `list<int32>` | Shape of the yield array |
| `yield_breakpoints` | `list<int32>` | Interpolation breakpoints |
| `yield_interpolation` | `list<int32>` | Interpolation codes |

### distributions.arrow

One row per angle-energy distribution.  Most columns are nullable -- only the
columns relevant to each distribution type are populated.

**Common columns:**

- `reaction_mt` (`int32`), `product_idx` (`int32`), `dist_idx` (`int32`)
- `type` (`utf8`): `"uncorrelated"`, `"correlated"`, `"kalbach-mann"`, or `"nbody"`
- `applicability_*`: Applicability tabulation (nullable)

**Uncorrelated angle columns** (nullable):
`angle_energies`, `angle_mu_data`, `angle_mu_offsets`,
`angle_mu_interpolation`

**Uncorrelated energy columns** (nullable):
`energy_dist_type`, `energy_dist_energies`, `energy_dist_interpolation`,
`energy_dist_data`, `energy_dist_offsets`, `energy_dist_out_interp`,
`energy_dist_n_discrete`, `energy_param_x`, `energy_param_y`,
`energy_param2_x`, `energy_param2_y`, `energy_restriction_u`,
`energy_threshold`, `energy_mass_ratio`, `energy_primary_flag`,
`energy_atomic_weight_ratio`, `energy_discrete_energy`

**Correlated columns** (nullable):
`corr_energies`, `corr_breakpoints`, `corr_interpolation`,
`corr_eout_data`, `corr_eout_offsets`, `corr_eout_interp`,
`corr_eout_n_discrete`, `corr_mu_data`, `corr_mu_offsets`,
`corr_mu_interp`

**Kalbach-Mann columns** (nullable):
`km_energies`, `km_breakpoints`, `km_interpolation`,
`km_data`, `km_offsets`, `km_interp`, `km_n_discrete`

**N-body columns** (nullable):
`nbody_n`, `nbody_total_mass`, `nbody_atomic_weight_ratio`,
`nbody_q_value`

### fast_xs.arrow

One row per temperature.  Contains the FastXSGrid lookup table for O(1)
cross-section retrieval during transport.

| Column | Type | Description |
|---|---|---|
| `temperature` | `utf8` | e.g. `"294K"` |
| `log_e_min` | `float64` | log of minimum energy |
| `inv_log_delta` | `float64` | Inverse of log bin width |
| `log_grid_index` | `list<int32>` | Grid index per log bin (8001 entries) |
| `xs` | `list<float64>` | Flattened (n_energy, 4) -- [total, absorption, scatter, fission] |
| `xs_shape` | `list<int32>` | `[n_energy, 4]` |
| `energy` | `list<float64>` | Full energy grid |
| `scatter_mt_numbers` | `list<int32>` | MT numbers for scattering channels |
| `scatter_mt_xs` | `list<float64>` | Flattened (n_energy, n_scatter) |
| `scatter_mt_shape` | `list<int32>` | Shape of scatter_mt_xs |
| `fission_mt_numbers` | `list<int32>` | MT numbers for fission channels |
| `fission_mt_xs` | `list<float64>` | Flattened (n_energy, n_fission) |
| `fission_mt_shape` | `list<int32>` | Shape of fission_mt_xs |
| `has_partial_fission` | `bool` | True if partial fission MTs present |
| `xs_ngamma` | `list<float64>` | (n,gamma) capture XS |
| `photon_prod` | `list<float64>` | Photon production XS |

### urr.arrow

One row per temperature.  Only present if unresolved resonance probability
tables exist.

| Column | Type | Description |
|---|---|---|
| `temperature` | `utf8` | e.g. `"294K"` |
| `energy` | `list<float64>` | URR energy grid |
| `table_data` | `list<float64>` | Flattened probability table (C order) |
| `table_shape` | `list<int32>` | `[n_energy, 6, n_band]` |
| `interpolation` | `int32` | 2 (lin-lin) or 5 (log-log) |
| `inelastic` | `int32` | Inelastic scattering flag |
| `absorption` | `int32` | Absorption flag |
| `multiply_smooth` | `bool` | Multiply-by-smooth-XS flag |

### total_nu.arrow

One row.  Only present for fissile nuclides with total nu data.

| Column | Type | Description |
|---|---|---|
| `particle` | `utf8` | `"neutron"` |
| `emission_mode` | `utf8` | `"prompt"`, `"delayed"`, or `"total"` |
| `decay_rate` | `float64` | Decay rate |
| `yield_type` | `utf8` | `"Tabulated1D"` or `"Polynomial"` |
| `yield_data` | `list<float64>` | Yield values |
| `yield_shape` | `list<int32>` | Shape |
| `yield_breakpoints` | `list<int32>` | Interpolation breakpoints |
| `yield_interpolation` | `list<int32>` | Interpolation codes |

### fission_photon.arrow

Two rows, one per `role`.  Only present for evaluations that carry a
fission energy release.  yamc reads this to scale fission photon
production; without it the release defaults to 1.0 and actinide photon
production comes out low.

The prompt and delayed terms are stored as the functions the evaluation
holds, so yamc evaluates them exactly rather than re-deriving them from
values on a grid.  The two terms are independent: U235, U238 and Pu239
store a tabulated prompt term alongside a polynomial delayed one, so
`kind` is per row.  A polynomial row fills `coefficients` and leaves the
table columns empty; a tabulated row fills `x`, `y`, `interpolation` and
`breakpoints` and leaves `coefficients` empty.

| Column | Type | Description |
|---|---|---|
| `role` | `utf8` | `"prompt_photons"` or `"delayed_photons"` |
| `kind` | `utf8` | `"polynomial"` or `"tabulated"` |
| `coefficients` | `list<float64>` | Ascending powers, for `"polynomial"` |
| `x` | `list<float64>` | Energy points, for `"tabulated"` |
| `y` | `list<float64>` | Release values, for `"tabulated"` |
| `interpolation` | `list<int32>` | Interpolation codes, lin-lin only |
| `breakpoints` | `list<int32>` | Interpolation region breakpoints |


### covariance.arrow

MF=33 cross-section covariance.  Optional, and absence is normal: data
published before this table existed has no such file, and a reader must treat a
missing `covariance.arrow` as "no covariance" rather than as an error.  The
matrices are large, so a consumer that does not need uncertainty pays nothing
for it.

One row per covariance block, which is one NC or NI sub-subsection of one MF=33
subsection.  No group structure is imposed: each block keeps the evaluation's
own energy grid, so `ne` differs from block to block and no common grid has to
exist.  That granularity is also what makes the table sparse — most (MT, MT1)
pairs have no cross terms and simply have no row.  Nothing is thresholded or
dropped, so a block is reproduced exactly as the tape gave it.

`mt` is the reaction the MF=33 section belongs to and `mt1`/`mat1` the reaction
it is correlated with.  The diagonal blocks are the rows with `mat1 == 0` and
`mt1` either `0` or equal to `mt`; only those are symmetric in themselves.  An
off-diagonal block's transpose is the (`mt1`, `mt`) block, not the block
itself.

`kind` selects which columns are populated, and rows are stored in tape order:
within a subsection the NC blocks come first, then the NI blocks, with
`block_idx` running across both.  That is what makes `(mt, subsection_idx,
block_idx)` a key — numbering the two lists separately would give an NC block
and an NI block the same one.

Written only when asked for.  `convert_neutron_xs` and
`convert_neutron_transport` take a `covariance` flag, off by default, so an
ordinary conversion produces a directory byte-identical to one from a build
without this section.  ENDF only: MF=33 is not carried through ACER, so asking
for covariance from an ACE table is an error rather than a silently empty file.
The Python converter does not write this section at all (issue #514); that
asymmetry is deliberate and is the only one in the format.

| Column | Type | Description |
|---|---|---|
| `mt` | `int32` | MT of the section this block belongs to |
| `subsection_idx` | `int32` | Position of the subsection within the section |
| `block_idx` | `int32` | Position of the block within the subsection, NC blocks first then NI, as one running index |
| `kind` | `utf8` | `"ni"` (given explicitly) or `"nc"` (derived from other reactions) |
| `mat1` | `int32` | MAT of the correlated reaction; 0 for this material |
| `mt1` | `int32` | MT of the correlated reaction |
| `xmf1`, `xlfs1` | `float64` | MF and final excited state of the correlated reaction |
| `mtl` | `int32` | MTL from the section HEAD: the MT this reaction is lumped into, 0 if none |

NI blocks (`kind = "ni"`), where `lb` selects the layout again:

| Column | Type | Description |
|---|---|---|
| `lb` | `int32` | Layout: 0–4, 5, 6, 8 or 9 |
| `ls` | `int32` | LB=5 symmetry flag.  1 = `fkk` is an upper triangle, transpose implied; 0 = `fkk` is a full asymmetric matrix |
| `lt`, `nt`, `np`, `ne`, `ner`, `nec` | `int32` | The tape's own counts, kept as declared |
| `ek`, `fk` | `list<float64>` | First (E, F) table; for LB=5, `ek` is the energy grid |
| `el`, `fl` | `list<float64>` | Second (E, F) table (LB 0–4) |
| `fkk` | `list<float64>` | LB=5 covariance matrix, in the format's packed order.  Expand it with `ls` and `ne`; it is never reshaped on write |
| `er`, `ec`, `fkl` | `list<float64>` | LB=6 row grid, column grid and matrix |

NC blocks (`kind = "nc"`), where `lty` selects the layout:

| Column | Type | Description |
|---|---|---|
| `lty` | `int32` | 0, or non-zero for a covariance derived from another material |
| `e1`, `e2` | `float64` | Energy range the derivation applies over |
| `nci` | `int32` | LTY=0 term count |
| `ci`, `xmti` | `list<float64>` | LTY=0 coefficients and their MTs |
| `mats`, `mts`, `nei` | `int32` | LTY≠0 material, reaction and weight count |
| `xmfs`, `xlfss` | `float64` | LTY≠0 MF and final excited state |
| `ei`, `wei` | `list<float64>` | LTY≠0 energies and weights |

## Photon data: `{element}.arrow/`

```text
Fe.arrow/
├── version.json
├── element.arrow
├── subshells.arrow
├── compton.arrow
└── bremsstrahlung.arrow
```

### version.json

Same structure as neutron version.json.

### element.arrow

One row of element-level data: metadata, union energy grid, main cross
sections (both linear and log-space), and form factors.

| Column | Type | Description |
|---|---|---|
| `name` | `utf8` | Element symbol |
| `Z` | `int32` | Atomic number |
| `ln_energy` | `list<float64>` | log of energy grid |
| `coherent_xs` | `list<float64>` | Coherent scattering cross section |
| `incoherent_xs` | `list<float64>` | Incoherent scattering cross section |
| `photoelectric_xs` | `list<float64>` | Photoelectric cross section |
| `pair_production_nuclear_xs` | `list<float64>` | Nuclear pair production cross section |
| `pair_production_electron_xs` | `list<float64>` | Electron pair production cross section |
| `heating_xs` | `list<float64>` | Heating cross section |
| `coherent_int_ff_x` / `_y` | `list<float64>` | Integrated coherent form factor |
| `coherent_ff_x` / `_y` | `list<float64>` | Coherent form factor |
| `coherent_anomalous_real_x` / `_y` | `list<float64>` | Real anomalous scattering factor |
| `coherent_anomalous_imag_x` / `_y` | `list<float64>` | Imaginary anomalous scattering factor |
| `incoherent_ff_x` / `_y` | `list<float64>` | Incoherent scattering function |

File-level metadata: `filetype=data_photon`, `version=4.0`

### subshells.arrow

One row per subshell.

| Column | Type | Description |
|---|---|---|
| `designator` | `utf8` | Subshell name (`"K"`, `"L1"`, etc.) |
| `binding_energy` | `float64` | Binding energy (eV) |
| `num_electrons` | `float64` | Number of electrons |
| `xs` | `list<float64>` | Photoionization cross section |
| `ln_xs` | `list<float64>` | log of photoionization XS |
| `threshold_idx` | `int32` | Index into the union energy grid |
| `transitions_data` | `list<float64>` | Flattened transition matrix (nullable) |
| `transitions_shape` | `list<int32>` | Shape of the transition matrix (nullable) |

### compton.arrow

One row.  Present when Compton profile data exists.  Includes pre-computed
CDFs from trapezoidal integration.

| Column | Type | Description |
|---|---|---|
| `num_electrons` | `list<float64>` | Electron occupancy per shell |
| `binding_energy` | `list<float64>` | Binding energies (eV) |
| `pz` | `list<float64>` | Electron momentum grid |
| `J_data` | `list<float64>` | Flattened Compton profiles (C order) |
| `J_shape` | `list<int32>` | `[n_shells, n_momentum]` |
| `J_cdf_data` | `list<float64>` | Flattened Compton profile CDFs (C order) |
| `J_cdf_shape` | `list<int32>` | `[n_shells, n_momentum]` |

### bremsstrahlung.arrow

One row.  Present when bremsstrahlung data exists.

| Column | Type | Description |
|---|---|---|
| `I` | `float64` | Mean excitation energy (eV) |
| `electron_energy` | `list<float64>` | Electron kinetic energies (eV) |
| `photon_energy` | `list<float64>` | Reduced photon energies |
| `num_electrons` | `list<float64>` | Subshell occupancies |
| `ionization_energy` | `list<float64>` | Ionization potentials (eV) |
| `dcs_data` | `list<float64>` | Flattened scaled DCS array (C order) |
| `dcs_shape` | `list<int32>` | `[n_electron_energies, n_photon_energies]` |


## Transmutation network: `transmutation_{library}.arrow/`

A transmutation network is stored as a root directory of independently
library-sourced *subsections*, each in its own subdirectory.  It is built
directly from ENDF decay, NFY and neutron source files, and split so a consumer
can assemble a chain from a different library per subsection (for example
decay, reactions and fission yields from ENDF/B-8.1 and isomeric branching from
TENDL-2025).

```text
transmutation_endfb-8.1.arrow/
├── manifest.json                 lists the subsections present + provenance
├── decay/
│   ├── nuclides.arrow            one row per nuclide (name, half_life, decay_energy)
│   ├── decay_modes.arrow         (optional: present when any nuclide decays)
│   ├── sources.arrow             (optional: decay photon/electron spectra)
│   └── provenance.json
├── reactions/
│   ├── reactions.arrow           transmutation reaction topology + Q
│   └── provenance.json
├── fission_yields/
│   ├── fission_yields.arrow      per fissioning parent
│   ├── aliases.arrow             (optional: nuclides that inherit a parent's yields)
│   └── provenance.json
└── branching/                    (reserved for isomeric branch ratios; not emitted yet)
```

Each subsection is self-describing: its primary table carries a `filetype`
metadata tag (`transmutation-decay`, `transmutation-reactions`,
`transmutation-fission_yields`, all `version=2.0`), and a `provenance.json`
records where it came from.  Tables within a subsection are joined back to the
primary table by the `nuclide` column; subsections are joined across by nuclide
`name`.

Notes on the split from format_version 1: `half_life` and `decay_energy` are
decay quantities and live in the decay subsection.  The derived count columns
and `has_fission_yields` flag from the old monolithic `nuclides.arrow` are gone
(a consumer recomputes them from the actual tables).  Fission-yield inheritance
moved from a `fission_yield_parent` column to `fission_yields/aliases.arrow`.
All reactions are always emitted (no subset).

### manifest.json

At the root.  Records `format_version` (currently `2`), `library`,
`converter_version`, `created_utc`, and a `subsections` object mapping each
emitted subsection name to its relative `path`.

### provenance.json (per subsection)

`subsection`, `library`, `source` (always `"endf"`), `converter_version`,
`created_utc`.  The reactions subsection additionally records
`branch_ratios_applied` (whether a `--branch-ratios` override was applied).

### decay/nuclides.arrow

One row per nuclide in the network.  Primary table of the decay subsection;
carries `filetype=transmutation-decay`.

| Column | Type | Description |
|---|---|---|
| `name` | `utf8` | Nuclide name in GNDS format (e.g. `"U235"`, `"Ag110_m1"`) |
| `half_life` | `float64` (nullable) | Half-life in seconds (null for stable nuclides) |
| `decay_energy` | `float64` | Total decay energy (eV) |
| `half_life_uncertainty` | `float64` (nullable) | Standard deviation on `half_life` (s). Null means the evaluation stated none, which is **not** zero |
| `decay_energy_uncertainty` | `float64` (nullable) | Standard deviation on `decay_energy` (eV), quadrature over the light/electromagnetic/heavy components. Null means the evaluation stated none |

The two uncertainty columns are nullable and appended last, so a file written
before they existed reads unchanged and a reader that does not know them finds
the columns it wants by name. Null and zero are different claims throughout:
MF=8 MT=457 says "not stated" by writing zero, so a zero there becomes null
here rather than an uncertainty measured to be exactly nought.

### decay/decay_modes.arrow

One row per decay mode.  Omitted when no nuclide has decay data.

| Column | Type | Description |
|---|---|---|
| `nuclide` | `utf8` | Parent nuclide name |
| `type` | `utf8` | Decay mode (e.g. `"beta-"`, `"alpha"`, `"ec/beta+"`) |
| `target` | `utf8` (nullable) | Daughter nuclide name (null for spontaneous fission) |
| `branching_ratio` | `float64` | Mode branching ratio |

### decay/sources.arrow

One row per decay-product source spectrum.  `Mixture` distributions are
flattened into one row per component, with the component probability multiplied
into the `intensities` column.  Omitted when no nuclide has decay sources.

| Column | Type | Description |
|---|---|---|
| `nuclide` | `utf8` | Emitting nuclide |
| `particle` | `utf8` | Emitted particle (`"photon"`, `"electron"`, ...) |
| `type` | `utf8` | `"discrete"` or `"tabular"` |
| `energies` | `list<float64>` | Energy grid (eV) |
| `intensities` | `list<float64>` | Intensities at each energy |

### reactions/reactions.arrow

One row per transmutation reaction (e.g. `(n,gamma)`, `(n,2n)`, `(n,p)`).
Primary table of the reactions subsection; carries
`filetype=transmutation-reactions`.

| Column | Type | Description |
|---|---|---|
| `nuclide` | `utf8` | Parent nuclide name |
| `type` | `utf8` | Reaction label (e.g. `"(n,gamma)"`) |
| `target` | `utf8` (nullable) | Product nuclide name |
| `Q` | `float64` | Q-value in eV |
| `branching_ratio` | `float64` | Branching ratio; reflects the `--branch-ratios` override when applied (see `provenance.branch_ratios_applied`). Isomeric branching from a dedicated library is a future overlay. |

### fission_yields/fission_yields.arrow

One row per (fissioning parent, incident energy) pair.  Only parents with their
own yield data appear here; inheritors are listed in `aliases.arrow`.  Primary
table of the fission_yields subsection; carries
`filetype=transmutation-fission_yields`.

| Column | Type | Description |
|---|---|---|
| `nuclide` | `utf8` | Fissioning parent |
| `energy` | `float64` | Incident neutron energy (eV) |
| `products` | `list<utf8>` | Fission product nuclide names |
| `yields` | `list<float64>` | Independent yields, aligned with `products` |

### fission_yields/aliases.arrow

Nuclides that inherit another parent's fission yields rather than carrying their
own.  Omitted when there are none.

| Column | Type | Description |
|---|---|---|
| `nuclide` | `utf8` | Nuclide that inherits yields |
| `fission_yield_parent` | `utf8` | Parent whose yields it uses |
