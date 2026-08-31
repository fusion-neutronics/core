"""Compare endf-python objects against Arrow readback for verification.

Uses tolerance-based comparison (np.allclose with rtol=1e-12) instead of
bitwise equality.  Verifies synthesized MTs and FastXSGrid consistency.
"""

from pathlib import Path

import numpy as np

from numpy.polynomial import Polynomial

from endf.function import Tabulated1D

from .neutron_reader import read_neutron_from_arrow
from .photon_reader import read_photon_from_arrow
from .synthesis import SYNTHETIC_MTS


def _arrays_close(a, b, label="", rtol=1e-12, atol=0.0):
    """Compare two arrays or lists for approximate equality."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        print(f"  MISMATCH {label}: shape {a.shape} != {b.shape}")
        return False
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        diff = np.abs(a - b)
        max_diff = np.max(diff)
        n_diff = np.sum(~np.isclose(a, b, rtol=rtol, atol=atol))
        print(f"  MISMATCH {label}: {n_diff} differing values, max diff = {max_diff}")
        return False
    return True


def _check_scalar(expected, actual, label=""):
    """Check scalar equality."""
    if expected != actual:
        print(f"  MISMATCH {label}: {expected!r} != {actual!r}")
        return False
    return True


def verify_neutron(data, arrow_path):
    """Verify that Arrow export matches the IncidentNeutron object.

    Uses tolerance-based comparison and verifies synthesized MTs and
    FastXSGrid consistency.

    Parameters
    ----------
    data : endf.IncidentNeutron
        Original data object.
    arrow_path : str or Path
        Path to the .arrow/ directory.

    Returns
    -------
    bool
        True if all data matches within tolerance.
    """
    arrow_path = Path(arrow_path)
    arrow_data = read_neutron_from_arrow(arrow_path)
    ok = True

    # --- version.json ---
    if "version" in arrow_data:
        v = arrow_data["version"]
        ok &= _check_scalar(1, v.get("format_version"), "format_version")
        if v.get("converter_version") is None:
            print("  WARNING: converter_version missing from version.json")

    # --- Nuclide metadata ---
    nuc = arrow_data["nuclide"]
    ok &= _check_scalar(data.name, nuc["name"], "name")
    ok &= _check_scalar(int(data.atomic_number), nuc["Z"], "Z")
    ok &= _check_scalar(int(data.mass_number), nuc["A"], "A")
    ok &= _check_scalar(
        float(data.atomic_weight_ratio), nuc["atomic_weight_ratio"], "AWR")

    # Temperatures
    ok &= _check_scalar(
        list(data.temperatures), nuc["temperatures"], "temperatures")

    # kTs
    expected_kts = [float(data.kTs[i]) for i in range(len(data.temperatures))]
    ok &= _arrays_close(expected_kts, nuc["kTs"], "kTs")

    # Energy grids
    energy_temps = nuc["energy_temperatures"]
    energy_vals = nuc["energy_values"]
    for i, t in enumerate(energy_temps):
        ok &= _arrays_close(
            data.energy[t], energy_vals[i], f"energy[{t}]")

    # --- Reactions (original) ---
    arrow_rxs = arrow_data["reactions"]
    arrow_rx_by_mt = {r["mt"]: r for r in arrow_rxs}

    for rx in data.reactions.values():
        mt = rx.MT
        if mt not in arrow_rx_by_mt:
            print(f"  MISMATCH: MT {mt} missing from Arrow reactions")
            ok = False
            continue

        arx = arrow_rx_by_mt[mt]
        ok &= _check_scalar(mt, arx["mt"], f"rx[{mt}].mt")
        ok &= _check_scalar(float(rx.q_reaction), arx["Q_value"], f"rx[{mt}].Q_value")
        ok &= _check_scalar(
            bool(rx.center_of_mass), arx["center_of_mass"],
            f"rx[{mt}].center_of_mass")

        # Cross sections
        for t_idx, T in enumerate(arx["xs_temperatures"]):
            if rx.xs.get(T) is not None:
                ok &= _arrays_close(
                    rx.xs[T].y, arx["xs_values"][t_idx],
                    f"rx[{mt}].xs[{T}]")
                expected_thresh = getattr(rx.xs[T], '_threshold_idx', 0)
                ok &= _check_scalar(
                    int(expected_thresh), arx["xs_threshold_idx"][t_idx],
                    f"rx[{mt}].xs_threshold[{T}]")

    # --- Synthesized MTs ---
    for mt in SYNTHETIC_MTS:
        if mt in arrow_rx_by_mt:
            arx = arrow_rx_by_mt[mt]
            if mt not in data.reactions:
                # This is a synthesized MT, verify it's marked redundant
                ok &= _check_scalar(True, arx["redundant"], f"synth_rx[{mt}].redundant")
                # Verify XS is not all zeros for MT 1 (total should always have data)
                if mt == 1:
                    for t_idx, T in enumerate(arx["xs_temperatures"]):
                        xs_arr = np.asarray(arx["xs_values"][t_idx], dtype=np.float64)
                        if np.all(xs_arr == 0) and len(xs_arr) > 0:
                            print(f"  WARNING: synth MT 1 is all zeros at {T}")
        elif mt == 1 or mt == 101:
            # MT 1 and 101 must always be present
            print(f"  MISMATCH: synthesized MT {mt} missing from reactions")
            ok = False

    # --- Products ---
    arrow_prods = arrow_data["products"]
    prod_idx = 0
    for rx in data.reactions.values():
        for p_idx, p in enumerate(rx.products):
            if prod_idx >= len(arrow_prods):
                print(f"  MISMATCH: ran out of arrow products at rx {rx.MT} p {p_idx}")
                ok = False
                continue
            ap = arrow_prods[prod_idx]
            ok &= _check_scalar(
                str(p.name), ap["particle"],
                f"product[{rx.MT},{p_idx}].particle")
            ok &= _check_scalar(
                str(p.emission_mode), ap["emission_mode"],
                f"product[{rx.MT},{p_idx}].emission_mode")
            ok &= _check_scalar(
                float(p.decay_rate), ap["decay_rate"],
                f"product[{rx.MT},{p_idx}].decay_rate")

            # Verify yield
            if isinstance(p.yield_, Tabulated1D):
                expected_data = np.concatenate([p.yield_.x, p.yield_.y])
                ok &= _arrays_close(
                    expected_data, ap["yield_data"],
                    f"product[{rx.MT},{p_idx}].yield_data")
            elif isinstance(p.yield_, Polynomial):
                ok &= _arrays_close(
                    p.yield_.coef, ap["yield_data"],
                    f"product[{rx.MT},{p_idx}].yield_data")

            prod_idx += 1

    # --- Distributions (spot check types) ---
    arrow_dists = arrow_data["distributions"]
    dist_idx = 0
    for rx in data.reactions.values():
        for p_idx, p in enumerate(rx.products):
            for d_idx, d in enumerate(p.distribution):
                if dist_idx >= len(arrow_dists):
                    print("  MISMATCH: ran out of arrow distributions")
                    ok = False
                    continue
                ad = arrow_dists[dist_idx]
                from endf.angle_energy import (
                    UncorrelatedAngleEnergy, CorrelatedAngleEnergy,
                    KalbachMann, NBodyPhaseSpace,
                )

                if isinstance(d, UncorrelatedAngleEnergy):
                    ok &= _check_scalar(
                        "uncorrelated", ad["type"],
                        f"dist[{rx.MT},{p_idx},{d_idx}].type")
                elif isinstance(d, CorrelatedAngleEnergy):
                    ok &= _check_scalar(
                        "correlated", ad["type"],
                        f"dist[{rx.MT},{p_idx},{d_idx}].type")
                elif isinstance(d, KalbachMann):
                    ok &= _check_scalar(
                        "kalbach-mann", ad["type"],
                        f"dist[{rx.MT},{p_idx},{d_idx}].type")
                elif isinstance(d, NBodyPhaseSpace):
                    ok &= _check_scalar(
                        "nbody", ad["type"],
                        f"dist[{rx.MT},{p_idx},{d_idx}].type")
                    ok &= _check_scalar(
                        int(d.n_particles), ad["nbody_n"],
                        f"dist[{rx.MT},{p_idx},{d_idx}].nbody_n")
                    ok &= _check_scalar(
                        float(d.total_mass), ad["nbody_total_mass"],
                        f"dist[{rx.MT},{p_idx},{d_idx}].nbody_total_mass")
                dist_idx += 1

    # --- FastXSGrid ---
    if "fast_xs" in arrow_data:
        for fxs in arrow_data["fast_xs"]:
            # Verify log_grid_index is monotonically non-decreasing
            lgi = np.asarray(fxs["log_grid_index"], dtype=np.int32)
            if not np.all(lgi[1:] >= lgi[:-1]):
                print(f"  MISMATCH fast_xs[{fxs['temperature']}]: "
                      "log_grid_index not monotonically non-decreasing")
                ok = False

            # Verify log_grid_index has N_LOG_BINS + 1 entries (8001)
            if len(lgi) != 8001:
                print(f"  MISMATCH fast_xs[{fxs['temperature']}]: "
                      f"log_grid_index has {len(lgi)} entries, expected 8001")
                ok = False

            # Verify xs consistency: total ≈ absorption + scattering + fission
            xs_shape = fxs["xs_shape"]
            xs = np.asarray(fxs["xs"], dtype=np.float64).reshape(xs_shape)
            total = xs[:, 0]
            absorption = xs[:, 1]
            scattering = xs[:, 2]
            fission = xs[:, 3]
            reconstructed = absorption + scattering + fission
            if not np.allclose(total, reconstructed, rtol=1e-10, atol=1e-30):
                n_bad = np.sum(~np.isclose(total, reconstructed, rtol=1e-10, atol=1e-30))
                print(f"  MISMATCH fast_xs[{fxs['temperature']}]: "
                      f"total != abs+scat+fis at {n_bad} points")
                ok = False
    else:
        print("  MISMATCH: fast_xs data missing")
        ok = False

    # --- URR ---
    if data.urr:
        if "urr" not in arrow_data:
            print("  MISMATCH: URR data missing from Arrow")
            ok = False
        else:
            arrow_urrs = arrow_data["urr"]
            for i, (temp, urr) in enumerate(data.urr.items()):
                au = arrow_urrs[i]
                ok &= _check_scalar(temp, au["temperature"], f"urr[{i}].temperature")
                ok &= _arrays_close(urr.energy, au["energy"], f"urr[{i}].energy")
                table_flat = np.asarray(urr.table).ravel(order='C')
                ok &= _arrays_close(
                    table_flat, au["table_data"], f"urr[{i}].table_data")
                ok &= _check_scalar(
                    int(urr.interpolation), au["interpolation"],
                    f"urr[{i}].interpolation")

    # --- total_nu ---
    total_nu_written = False
    for rx in data.reactions.values():
        if len(rx.derived_products) > 0 and not total_nu_written:
            dp = rx.derived_products[0]
            total_nu_written = True
            if "total_nu" not in arrow_data:
                print("  MISMATCH: total_nu missing from Arrow")
                ok = False
            else:
                tnu = arrow_data["total_nu"]
                ok &= _check_scalar(
                    str(dp.name), tnu["particle"], "total_nu.particle")
                if isinstance(dp.yield_, Tabulated1D):
                    expected_data = np.concatenate([dp.yield_.x, dp.yield_.y])
                    ok &= _arrays_close(
                        expected_data, tnu["yield_data"], "total_nu.yield_data")
                elif isinstance(dp.yield_, Polynomial):
                    ok &= _arrays_close(
                        dp.yield_.coef, tnu["yield_data"], "total_nu.yield_data")

    if ok:
        print("Neutron verification PASSED")
    else:
        print("Neutron verification FAILED")
    return ok


def verify_photon(data, arrow_path):
    """Verify that Arrow export matches the IncidentPhoton object.

    Uses tolerance-based comparison.

    Parameters
    ----------
    data : endf.IncidentPhoton
        Original data object.
    arrow_path : str or Path
        Path to the .arrow/ directory.

    Returns
    -------
    bool
        True if all data matches within tolerance.
    """
    arrow_path = Path(arrow_path)
    arrow_data = read_photon_from_arrow(arrow_path)
    ok = True

    elem = arrow_data["element"]
    ok &= _check_scalar(data.name, elem["name"], "name")
    ok &= _check_scalar(int(data.atomic_number), elem["Z"], "Z")

    # Union energy grid. Only the log form ships: the raw `energy` column was
    # retired (#500), so `ln_energy` below is what verifies the grid.
    union_grid = np.array([])
    for rx in data:
        union_grid = np.union1d(union_grid, rx.xs.x)

    # Verify log-space energy
    if elem.get("ln_energy") is not None and len(elem["ln_energy"]) > 0:
        expected_ln_energy = np.log(union_grid)
        ok &= _arrays_close(expected_ln_energy, elem["ln_energy"], "ln_energy")

    # Main cross sections
    from endf.incident_photon import PHOTON_REACTION_NAME
    for mt, rx in data.reactions.items():
        key = PHOTON_REACTION_NAME[mt]
        if mt in (502, 504, 515, 517, 522, 525):
            expected_xs = np.asarray(rx.xs(union_grid), dtype=np.float64)
            col_map = {
                "coherent": "coherent_xs",
                "incoherent": "incoherent_xs",
                "photoelectric": "photoelectric_xs",
                "pair_production_nuclear": "pair_production_nuclear_xs",
                "pair_production_electron": "pair_production_electron_xs",
                "heating": "heating_xs",
            }
            if key in col_map:
                ok &= _arrays_close(
                    expected_xs, elem[col_map[key]], f"xs[{key}]")

    # Subshells
    if arrow_data["subshells"]:
        for i, sub in enumerate(arrow_data["subshells"]):
            desig = sub["designator"]
            for mt, rx in data.reactions.items():
                rkey = PHOTON_REACTION_NAME[mt]
                if rkey == desig and mt >= 534 and mt <= 572:
                    threshold = rx.xs.x[0]
                    idx = int(np.searchsorted(union_grid, threshold, side='right') - 1)
                    expected_xs = np.asarray(
                        rx.xs(union_grid[idx:]), dtype=np.float64)
                    ok &= _arrays_close(
                        expected_xs, sub["xs"], f"subshell[{desig}].xs")
                    ok &= _check_scalar(
                        idx, sub["threshold_idx"],
                        f"subshell[{desig}].threshold_idx")
                    break

    # Compton profiles
    if data.compton_profiles and "compton" in arrow_data:
        profile = data.compton_profiles
        cmp = arrow_data["compton"]
        ok &= _arrays_close(
            profile['num_electrons'], cmp["num_electrons"], "compton.num_electrons")
        ok &= _arrays_close(
            profile['binding_energy'], cmp["binding_energy"], "compton.binding_energy")
        ok &= _arrays_close(
            profile['J'][0].x, cmp["pz"], "compton.pz")
        J_arr = np.array([Jk.y for Jk in profile['J']])
        ok &= _arrays_close(
            J_arr.ravel(order='C'), cmp["J_data"], "compton.J_data")

        # Verify CDFs exist
        if "J_cdf_data" in cmp and cmp["J_cdf_data"] is not None:
            cdf_arr = np.asarray(cmp["J_cdf_data"], dtype=np.float64)
            cdf_shape = cmp["J_cdf_shape"]
            if len(cdf_arr) > 0:
                cdf = cdf_arr.reshape(cdf_shape)
                # CDFs should be monotonically non-decreasing per shell
                for s in range(cdf.shape[0]):
                    if not np.all(cdf[s, 1:] >= cdf[s, :-1] - 1e-15):
                        print(f"  MISMATCH: compton CDF shell {s} not monotonic")
                        ok = False

    # Bremsstrahlung
    if data.bremsstrahlung and "bremsstrahlung" in arrow_data:
        brem = data.bremsstrahlung
        abrem = arrow_data["bremsstrahlung"]
        ok &= _check_scalar(float(brem['I']), abrem["I"], "brem.I")
        ok &= _arrays_close(
            brem['electron_energy'], abrem["electron_energy"],
            "brem.electron_energy")
        ok &= _arrays_close(
            brem['photon_energy'], abrem["photon_energy"],
            "brem.photon_energy")
        dcs_arr = np.asarray(brem['dcs'])
        ok &= _arrays_close(
            dcs_arr.ravel(order='C'), abrem["dcs_data"], "brem.dcs_data")

    if ok:
        print("Photon verification PASSED")
    else:
        print("Photon verification FAILED")
    return ok
