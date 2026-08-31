"""Synthesize hierarchical MTs and build FastXSGrid lookup tables.

This module implements yamc's post-processing algorithms in Python so that
the Arrow files contain simulation-ready data.
"""

import numpy as np

from endf.data import SUM_RULES


# MT numbers that count as scattering (non-inelastic subset).
#
# This is yamc's own classification for the per-channel scatter breakdown in
# fast_xs, not an ENDF summation rule, so it is defined here rather than taken
# from SUM_RULES. Every channel that re-emits a neutron belongs in it.
#
# The 152-200 range is a convenient way to reach the multi-particle channels, but
# it also sweeps up the few in that range that emit no neutron at all: (n,ta),
# (n,dt), (n,p3He), (n,d3He), (n,3Hea) and (n,3p). Those are disappearance
# reactions and belong to MT 101 instead. Since MT 3 is built as
# scattering + fission + MT 101, leaving them in both groups counts them twice,
# so they are subtracted out. See issue #23.
_SCATTERING_CANDIDATES = frozenset([
    2, 5, 11, 16, 17, 22, 23, 24, 25, 28, 29, 30,
    32, 33, 34, 35, 36, 37, 41, 42, 44, 45,
    *range(152, 201),
    *range(875, 891),
])
SCATTERING_MTS_NON_INELASTIC = _SCATTERING_CANDIDATES - frozenset(SUM_RULES[101])

# Inelastic scattering: MT 50-91
INELASTIC_MTS = frozenset(SUM_RULES[4])

# All scattering MTs
ALL_SCATTERING_MTS = INELASTIC_MTS | SCATTERING_MTS_NON_INELASTIC

# Fission MTs: the total (18) and its partials
FISSION_MTS = frozenset([18, *SUM_RULES[18]])

# MTs that contribute to MT 101, neutron disappearance.
#
# Taken from the ENDF summation rules rather than spelled out here. A private
# copy previously included 13 channels that re-emit a neutron, such as (n,npd),
# (n,2n2p) and (n,n3p), which do not belong in a disappearance sum. See #23.
ABSORPTION_MTS = frozenset(SUM_RULES[101])

# MT 3 is built as scattering + fission + MT 101, so the three groups have to be
# disjoint or a channel is counted twice. Asserted here so any future change to
# the sets fails loudly rather than quietly inflating MT 3 and MT 1.
assert not (ABSORPTION_MTS & (INELASTIC_MTS | SCATTERING_MTS_NON_INELASTIC))
assert not (ABSORPTION_MTS & FISSION_MTS)
assert not ((INELASTIC_MTS | SCATTERING_MTS_NON_INELASTIC) & FISSION_MTS)

# Synthesized MT numbers -- these are redundant sums
SYNTHETIC_MTS = frozenset([1, 3, 4, 27, 101])

# Number of log bins for FastXSGrid
N_LOG_BINS = 8000


def is_scattering_mt(mt):
    """Return True if the given MT is a scattering reaction."""
    return mt in ALL_SCATTERING_MTS


def is_fission_mt(mt):
    """Return True if the given MT is a fission reaction."""
    return mt in FISSION_MTS


def _interp_xs_to_grid(reaction, temperature, energy_grid):
    """Interpolate a reaction's cross section onto the full energy grid.

    Parameters
    ----------
    reaction : endf.Reaction
        The reaction object.
    temperature : str
        Temperature key (e.g., "294K").
    energy_grid : numpy.ndarray
        The full energy grid for this temperature.

    Returns
    -------
    numpy.ndarray
        Cross-section values on the full energy grid.
    """
    xs_tab = reaction.xs.get(temperature)
    if xs_tab is None:
        return np.zeros(len(energy_grid), dtype=np.float64)

    # xs_tab is a Tabulated1D-like with .x, .y and a _threshold_idx
    threshold_idx = getattr(xs_tab, '_threshold_idx', 0)
    xs_y = np.asarray(xs_tab.y, dtype=np.float64)

    # Build result array (zeros below threshold)
    result = np.zeros(len(energy_grid), dtype=np.float64)

    if len(xs_y) == 0:
        return result

    # The xs_y array starts at threshold_idx into the energy grid
    n_xs = len(xs_y)
    end_idx = threshold_idx + n_xs
    if end_idx <= len(energy_grid):
        result[threshold_idx:end_idx] = xs_y
    else:
        # Clip to available grid length
        avail = len(energy_grid) - threshold_idx
        result[threshold_idx:] = xs_y[:avail]

    return result


def synthesize_hierarchical_mts(data, temperature):
    """Synthesize hierarchical MT cross sections from constituent reactions.

    Parameters
    ----------
    data : endf.IncidentNeutron
        The incident neutron data object.
    temperature : str
        Temperature key (e.g., "294K").

    Returns
    -------
    dict
        Mapping of MT number to numpy cross-section array on the full energy
        grid.  Keys are 4, 101, 27, 3, 1.  MT 1 and 101 are always present
        (even if all zeros).
    """
    energy_grid = np.asarray(data.energy[temperature], dtype=np.float64)
    n_energy = len(energy_grid)

    # Pre-interpolate all reactions onto the full grid
    rx_xs = {}
    for mt, rx in data.reactions.items():
        if mt not in SYNTHETIC_MTS:
            rx_xs[mt] = _interp_xs_to_grid(rx, temperature, energy_grid)

    result = {}

    # MT 4 = sum(MT 50-91) -- total inelastic
    mt4 = np.zeros(n_energy, dtype=np.float64)
    for mt in INELASTIC_MTS:
        if mt in rx_xs:
            mt4 += rx_xs[mt]
    result[4] = mt4

    # MT 101 = sum(MT 102-117, 155, 182-199) -- absorption excluding fission
    mt101 = np.zeros(n_energy, dtype=np.float64)
    for mt in ABSORPTION_MTS:
        if mt in rx_xs:
            mt101 += rx_xs[mt]
    result[101] = mt101

    # MT 18 total fission (use if present, otherwise sum partial fission)
    fission_xs = np.zeros(n_energy, dtype=np.float64)
    if 18 in rx_xs:
        fission_xs = rx_xs[18].copy()
    else:
        for mt in [19, 20, 21, 38]:
            if mt in rx_xs:
                fission_xs += rx_xs[mt]

    # MT 27 = MT 18 + MT 101 -- total absorption
    result[27] = fission_xs + mt101

    # MT 3 = non-elastic (all scattering except elastic + fission + absorption)
    # = sum of all non-synthetic, non-redundant reactions that are scattering
    #   + fission + MT 101
    mt3 = np.zeros(n_energy, dtype=np.float64)
    for mt, xs in rx_xs.items():
        if mt in SYNTHETIC_MTS:
            continue
        if is_scattering_mt(mt) and mt != 2:
            mt3 += xs
    mt3 += fission_xs + mt101
    result[3] = mt3

    # MT 1 = MT 2 (elastic) + MT 3 (non-elastic) -- total
    elastic = rx_xs.get(2, np.zeros(n_energy, dtype=np.float64))
    result[1] = elastic + mt3

    return result


def build_fast_xs(data, temperature, synthetic_mts):
    """Build FastXSGrid arrays for fast cross-section lookup.

    Parameters
    ----------
    data : endf.IncidentNeutron
        The incident neutron data object.
    temperature : str
        Temperature key (e.g., "294K").
    synthetic_mts : dict
        Mapping from MT number to numpy XS arrays (from
        ``synthesize_hierarchical_mts``).

    Returns
    -------
    dict
        Dictionary with FastXSGrid data:
        - ``log_e_min`` (float): log of minimum energy
        - ``inv_log_delta`` (float): inverse of log bin width
        - ``log_grid_index`` (ndarray): grid index per log bin (N_LOG_BINS+1)
        - ``xs`` (ndarray): shape (n_energy, 4) -- [total, absorption, scatter, fission]
        - ``energy`` (ndarray): the full energy grid
        - ``scatter_mt_numbers`` (list of int): MT numbers for scattering channels
        - ``scatter_mt_xs`` (ndarray): shape (n_energy, n_scatter) -- XS per scatter MT
        - ``fission_mt_numbers`` (list of int): MT numbers for fission channels
        - ``fission_mt_xs`` (ndarray): shape (n_energy, n_fission) -- XS per fission MT
        - ``has_partial_fission`` (bool): True if partial fission MTs present
        - ``xs_ngamma`` (ndarray): (n,gamma) capture XS (MT 102)
        - ``photon_prod`` (ndarray): photon production XS
    """
    energy_grid = np.asarray(data.energy[temperature], dtype=np.float64)
    n_energy = len(energy_grid)

    # Build the 4-column XS array: [total, absorption, scatter, fission]
    mt1 = synthetic_mts.get(1, np.zeros(n_energy, dtype=np.float64))
    mt101 = synthetic_mts.get(101, np.zeros(n_energy, dtype=np.float64))

    # Fission XS
    fission_xs = np.zeros(n_energy, dtype=np.float64)
    rx_xs_cache = {}
    for mt, rx in data.reactions.items():
        if mt not in SYNTHETIC_MTS:
            rx_xs_cache[mt] = _interp_xs_to_grid(rx, temperature, energy_grid)

    if 18 in rx_xs_cache:
        fission_xs = rx_xs_cache[18].copy()
    else:
        for mt in [19, 20, 21, 38]:
            if mt in rx_xs_cache:
                fission_xs += rx_xs_cache[mt]

    # Absorption = MT 101 (disappearance, excluding fission)
    # Fission is stored as a separate column
    absorption = mt101.copy()

    # Scattering = total - absorption - fission
    scattering = mt1 - absorption - fission_xs
    # Clamp to zero (numerical noise)
    scattering = np.maximum(scattering, 0.0)

    xs_4col = np.column_stack([mt1, absorption, scattering, fission_xs])

    # Build log grid index
    log_e_min = np.log(energy_grid[0]) if energy_grid[0] > 0 else -100.0
    log_e_max = np.log(energy_grid[-1]) if energy_grid[-1] > 0 else 100.0
    log_delta = (log_e_max - log_e_min) / N_LOG_BINS
    inv_log_delta = 1.0 / log_delta if log_delta > 0 else 0.0

    log_grid_index = np.zeros(N_LOG_BINS + 1, dtype=np.int32)
    log_energy = np.log(energy_grid)

    for i in range(N_LOG_BINS + 1):
        log_e = log_e_min + i * log_delta
        # Find the index in energy_grid where log(E) >= log_e
        idx = int(np.searchsorted(log_energy, log_e, side='right')) - 1
        log_grid_index[i] = max(0, min(idx, n_energy - 1))

    # Scatter MT breakdown
    scatter_mt_numbers = []
    scatter_mt_xs_list = []
    for mt in sorted(rx_xs_cache.keys()):
        if is_scattering_mt(mt):
            scatter_mt_numbers.append(mt)
            scatter_mt_xs_list.append(rx_xs_cache[mt])

    if scatter_mt_xs_list:
        scatter_mt_xs = np.column_stack(scatter_mt_xs_list)
    else:
        scatter_mt_xs = np.empty((n_energy, 0), dtype=np.float64)

    # Fission MT breakdown
    fission_mt_numbers = []
    fission_mt_xs_list = []
    has_partial_fission = False
    for mt in [19, 20, 21, 38]:
        if mt in rx_xs_cache:
            fission_mt_numbers.append(mt)
            fission_mt_xs_list.append(rx_xs_cache[mt])
            has_partial_fission = True

    if not has_partial_fission and 18 in rx_xs_cache:
        fission_mt_numbers = [18]
        fission_mt_xs_list = [rx_xs_cache[18]]

    if fission_mt_xs_list:
        fission_mt_xs = np.column_stack(fission_mt_xs_list)
    else:
        fission_mt_xs = np.empty((n_energy, 0), dtype=np.float64)

    # (n,gamma) capture -- MT 102
    xs_ngamma = rx_xs_cache.get(102, np.zeros(n_energy, dtype=np.float64))

    # Photon production XS -- sum of all photon-producing reaction XS
    # For simplicity, use zeros; yamc computes this from product data
    photon_prod = np.zeros(n_energy, dtype=np.float64)

    return {
        "log_e_min": float(log_e_min),
        "inv_log_delta": float(inv_log_delta),
        "log_grid_index": log_grid_index,
        "xs": xs_4col,
        "energy": energy_grid,
        "scatter_mt_numbers": scatter_mt_numbers,
        "scatter_mt_xs": scatter_mt_xs,
        "fission_mt_numbers": fission_mt_numbers,
        "fission_mt_xs": fission_mt_xs,
        "has_partial_fission": has_partial_fission,
        "xs_ngamma": xs_ngamma,
        "photon_prod": photon_prod,
    }
