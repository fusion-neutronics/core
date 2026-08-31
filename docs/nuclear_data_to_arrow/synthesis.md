# MT synthesis and FastXSGrid

The `synthesis` module implements yamc's post-processing algorithms in Python.
These run at conversion time so the Arrow files are simulation-ready.

## Hierarchical MT synthesis

ENDF evaluations store individual reaction cross sections (e.g., MT 102 for
radiative capture, MT 51-91 for inelastic levels).  For transport, yamc needs
aggregate cross sections that sum these into hierarchical totals:

| Synthesized MT | Name | Definition |
|---|---|---|
| **MT 4** | Total inelastic | sum of MT 50–91 |
| **MT 101** | Disappearance | sum of MT 102–117, 155, 182–199 |
| **MT 27** | Total absorption | MT 18 (fission) + MT 101 |
| **MT 3** | Non-elastic | all scattering (except elastic) + fission + MT 101 |
| **MT 1** | Total | MT 2 (elastic) + MT 3 |

MT 1 and MT 101 are always created, even if all constituent cross sections
are zero.

### Which MTs are scattering?

Scattering MTs include:

- **Inelastic**: MT 50–91
- **Non-inelastic scattering**: MT 2, 5, 11, 16, 17, 22–25, 28–30, 32–37,
  41–42, 44–45, 152–200, 875–890

Fission MTs (18, 19, 20, 21, 38) and absorption MTs (102–117, 155, 182–199)
are explicitly excluded from the scattering category.

### Algorithm

1. All reaction cross sections are interpolated onto the full energy grid
   for the given temperature.
2. Synthesized MTs are computed as sums of their constituents.
3. The synthesized XS arrays are added to `reactions.arrow` with
   `redundant=True`.

## FastXSGrid

The FastXSGrid provides O(1) cross-section lookup by energy during particle
transport.  It replaces binary search over the energy grid with a log-binned
index.

### Structure

For each temperature, `fast_xs.arrow` stores:

`log_grid_index`
: An array of 8001 integers (for 8000 log-spaced bins).  Given a particle
  energy *E*, compute `bin = (ln(E) - log_e_min) * inv_log_delta` to get
  the starting index into the energy grid.

`xs`
: A `(n_energy, 4)` array with columns:
  - `[0]` total (MT 1)
  - `[1]` absorption (MT 101, excluding fission)
  - `[2]` scattering (total - absorption - fission)
  - `[3]` fission (MT 18 or sum of partial fission MTs)

  The invariant `total = absorption + scattering + fission` holds at every
  energy point.

`scatter_mt_xs` / `fission_mt_xs`
: Per-channel breakdowns so the transport code can sample individual
  scattering or fission reactions.

`xs_ngamma`
: The (n,gamma) capture cross section (MT 102), stored separately for
  common use in tallying.

### Log grid construction

```text
log_e_min = ln(E_grid[0])
log_e_max = ln(E_grid[-1])
log_delta = (log_e_max - log_e_min) / 8000
inv_log_delta = 1.0 / log_delta

For bin i = 0..8000:
    log_e = log_e_min + i * log_delta
    log_grid_index[i] = index in E_grid where ln(E) >= log_e
```

The resulting index is monotonically non-decreasing, which the verification
step checks.
