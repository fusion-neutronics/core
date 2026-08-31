"""Export an IncidentNeutron object to a simulation-ready .arrow/ directory."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from numpy.polynomial import Polynomial

from endf.function import Tabulated1D
from endf.reaction import REACTION_NAME

from .completion import write_completion_marker
from .schemas import (
    NUCLIDE_SCHEMA,
    REACTIONS_SCHEMA,
    PRODUCTS_SCHEMA,
    DISTRIBUTIONS_SCHEMA,
    URR_SCHEMA,
    FISSION_PHOTON_SCHEMA,
    TOTAL_NU_SCHEMA,
    FAST_XS_SCHEMA,
)
from .synthesis import (
    SYNTHETIC_MTS,
    synthesize_hierarchical_mts,
    build_fast_xs,
)


# ENDF interpolation scheme code for linear-linear. Every published
# fission_energy_release table uses this and nothing else.
_ENDF_LIN_LIN = 2


def _fission_photon_rows(data):
    """Rows for `fission_photon.arrow`, or `[]` when the evaluation has none.

    Emits the prompt and delayed photon energy release as the functions the
    evaluation stores, so yamc evaluates them exactly rather than re-deriving them
    from values on a grid (yamc issue #369).

    Raises on any form yamc cannot evaluate exactly. That is deliberate: a
    silently mis-evaluated release would bias fission photon production by tens of
    percent with nothing to show for it, which is how the missing scaling went
    unnoticed in the first place. Failing the conversion is loud and fixable.
    """
    release = getattr(data, "fission_energy", None)
    if release is None:
        return []

    rows = []
    for role in ("prompt_photons", "delayed_photons"):
        func = getattr(release, role, None)
        if func is None:
            raise ValueError(
                f"{data.name}: fission_energy_release is present but {role} is "
                f"missing; both terms are needed to form the scaling"
            )

        row = {
            "role": role,
            "kind": None,
            "coefficients": [],
            "x": [],
            "y": [],
            "interpolation": [],
            "breakpoints": [],
        }

        if isinstance(func, Polynomial):
            row["kind"] = "polynomial"
            # numpy Polynomial stores ascending powers, which is what yamc's
            # Horner evaluation expects.
            row["coefficients"] = np.asarray(func.coef, dtype=np.float64).tolist()
        elif isinstance(func, Tabulated1D):
            interpolation = np.asarray(func.interpolation, dtype=np.int32)
            breakpoints = np.asarray(func.breakpoints, dtype=np.int32)
            bad = sorted(set(interpolation.tolist()) - {_ENDF_LIN_LIN})
            if bad:
                raise ValueError(
                    f"{data.name} {role}: fission energy release table uses ENDF "
                    f"interpolation scheme(s) {bad}, but yamc only evaluates "
                    f"{_ENDF_LIN_LIN} (linear-linear) exactly. Emitting it anyway "
                    f"would silently bias fission photon production; add support "
                    f"for the scheme on both sides instead (yamc issue #369)."
                )
            if len(breakpoints) > 1:
                raise ValueError(
                    f"{data.name} {role}: fission energy release table is "
                    f"multi-region (breakpoints {breakpoints.tolist()}), which yamc "
                    f"does not evaluate. Add multi-region support on both sides "
                    f"rather than flattening it (yamc issue #369)."
                )
            row["kind"] = "tabulated"
            row["x"] = np.asarray(func.x, dtype=np.float64).tolist()
            row["y"] = np.asarray(func.y, dtype=np.float64).tolist()
            row["interpolation"] = interpolation.tolist()
            row["breakpoints"] = breakpoints.tolist()
        else:
            raise ValueError(
                f"{data.name} {role}: fission energy release is a "
                f"{type(func).__name__}, which yamc cannot evaluate. Only "
                f"Polynomial and Tabulated1D are supported (yamc issue #369)."
            )

        rows.append(row)

    return rows


def _serialize_yield(yield_obj):
    """Serialize a Function1D yield to (type, data, shape, breakpoints, interpolation)."""
    if isinstance(yield_obj, Tabulated1D):
        data = np.concatenate([yield_obj.x, yield_obj.y]).tolist()
        shape = [2, len(yield_obj.x)]
        breakpoints = list(int(b) for b in yield_obj.breakpoints)
        interpolation = list(int(i) for i in yield_obj.interpolation)
        return "Tabulated1D", data, shape, breakpoints, interpolation
    elif isinstance(yield_obj, Polynomial):
        data = list(float(c) for c in yield_obj.coef)
        shape = [len(yield_obj.coef)]
        return "Polynomial", data, shape, [], []
    else:
        raise TypeError(f"Unsupported yield type: {type(yield_obj)}")


def _serialize_tabulated1d(tab):
    """Serialize a Tabulated1D to (x, y, breakpoints, interpolation)."""
    return (
        np.asarray(tab.x, dtype=np.float64),
        np.asarray(tab.y, dtype=np.float64),
        np.asarray(tab.breakpoints, dtype=np.int32),
        np.asarray(tab.interpolation, dtype=np.int32),
    )


def _serialize_applicability(applicability_obj):
    """Serialize an applicability Tabulated1D."""
    if applicability_obj is None:
        return None, None, None, None
    x, y, bp, interp = _serialize_tabulated1d(applicability_obj)
    data = np.concatenate([x, y]).tolist()
    shape = [2, len(x)]
    return data, shape, bp.tolist(), interp.tolist()


def _serialize_angle_distribution(angle_dist):
    """Serialize an AngleDistribution to flat arrays.

    Returns (energies, mu_data, offsets, interpolation) or all-None.
    """
    if angle_dist is None:
        return None, None, None, None

    energies = np.asarray(angle_dist.energy, dtype=np.float64)

    # Convert all mu distributions to tabular form
    from endf.univariate import Tabular, Discrete
    mu_tabular = []
    for mu_i in angle_dist.mu:
        if isinstance(mu_i, (Tabular, Discrete)):
            mu_tabular.append(mu_i)
        else:
            mu_tabular.append(mu_i.to_tabular())

    n_pairs = sum(len(mu_i.x) for mu_i in mu_tabular)
    pairs = np.empty((3, n_pairs), dtype=np.float64)
    offsets = np.empty(len(mu_tabular), dtype=np.int32)
    interpolation = np.empty(len(mu_tabular), dtype=np.int32)
    j = 0
    for i, mu_i in enumerate(mu_tabular):
        n = len(mu_i.x)
        offsets[i] = j
        if isinstance(mu_i, Discrete):
            interpolation[i] = 1
        else:
            interpolation[i] = 1 if mu_i.interpolation == 'histogram' else 2
        pairs[0, j:j+n] = mu_i.x
        pairs[1, j:j+n] = mu_i.p
        pairs[2, j:j+n] = mu_i.c
        j += n

    mu_data = pairs.ravel(order='C').tolist()
    return energies.tolist(), mu_data, offsets.tolist(), interpolation.tolist()


def _serialize_energy_distribution(energy_dist):
    """Serialize an EnergyDistribution subclass into a dict of column values."""
    from endf.mf5 import (
        ContinuousTabular, MaxwellEnergy, WattEnergy, Evaporation,
        LevelInelastic, DiscretePhoton, MadlandNix,
    )

    result = {}

    if isinstance(energy_dist, ContinuousTabular):
        result["energy_dist_type"] = "continuous"
        result["energy_dist_energies"] = np.asarray(energy_dist.energy, dtype=np.float64).tolist()
        interp_flat = np.concatenate([
            np.asarray(energy_dist.breakpoints, dtype=np.int32),
            np.asarray(energy_dist.interpolation, dtype=np.int32),
        ]).tolist()
        result["energy_dist_interpolation"] = interp_flat

        from endf.univariate import Tabular, Discrete, Mixture
        n_pairs = sum(len(d) for d in energy_dist.energy_out)
        pairs = np.empty((3, n_pairs), dtype=np.float64)
        offsets = np.empty(len(energy_dist.energy_out), dtype=np.int32)
        out_interp = np.empty(len(energy_dist.energy_out), dtype=np.int32)
        n_discrete = np.empty(len(energy_dist.energy_out), dtype=np.int32)
        j = 0
        for i, eout in enumerate(energy_dist.energy_out):
            n = len(eout)
            offsets[i] = j
            if isinstance(eout, Mixture):
                discrete, continuous = eout.distribution
                m = len(discrete)
                n_discrete[i] = m
                out_interp[i] = 1 if continuous.interpolation == 'histogram' else 2
                pairs[0, j:j+m] = discrete.x
                pairs[1, j:j+m] = discrete.p
                pairs[2, j:j+m] = discrete.c
                pairs[0, j+m:j+n] = continuous.x
                pairs[1, j+m:j+n] = continuous.p
                pairs[2, j+m:j+n] = continuous.c
            else:
                if isinstance(eout, Tabular):
                    n_discrete[i] = 0
                    out_interp[i] = 1 if eout.interpolation == 'histogram' else 2
                elif isinstance(eout, Discrete):
                    n_discrete[i] = n
                    out_interp[i] = 1
                pairs[0, j:j+n] = eout.x
                pairs[1, j:j+n] = eout.p
                pairs[2, j:j+n] = eout.c
            j += n
        result["energy_dist_data"] = pairs.ravel(order='C').tolist()
        result["energy_dist_offsets"] = offsets.tolist()
        result["energy_dist_out_interp"] = out_interp.tolist()
        result["energy_dist_n_discrete"] = n_discrete.tolist()

    elif isinstance(energy_dist, MaxwellEnergy):
        result["energy_dist_type"] = "maxwell"
        result["energy_restriction_u"] = float(energy_dist.u)
        x, y, _, _ = _serialize_tabulated1d(energy_dist.theta)
        result["energy_param_x"] = x.tolist()
        result["energy_param_y"] = y.tolist()

    elif isinstance(energy_dist, Evaporation):
        result["energy_dist_type"] = "evaporation"
        result["energy_restriction_u"] = float(energy_dist.u)
        x, y, _, _ = _serialize_tabulated1d(energy_dist.theta)
        result["energy_param_x"] = x.tolist()
        result["energy_param_y"] = y.tolist()

    elif isinstance(energy_dist, WattEnergy):
        result["energy_dist_type"] = "watt"
        result["energy_restriction_u"] = float(energy_dist.u)
        x, y, _, _ = _serialize_tabulated1d(energy_dist.a)
        result["energy_param_x"] = x.tolist()
        result["energy_param_y"] = y.tolist()
        x2, y2, _, _ = _serialize_tabulated1d(energy_dist.b)
        result["energy_param2_x"] = x2.tolist()
        result["energy_param2_y"] = y2.tolist()

    elif isinstance(energy_dist, LevelInelastic):
        result["energy_dist_type"] = "level"
        result["energy_threshold"] = float(energy_dist.threshold)
        result["energy_mass_ratio"] = float(energy_dist.mass_ratio)

    elif isinstance(energy_dist, DiscretePhoton):
        result["energy_dist_type"] = "discrete_photon"
        result["energy_primary_flag"] = int(energy_dist.primary_flag)
        result["energy_discrete_energy"] = float(energy_dist.energy)
        result["energy_atomic_weight_ratio"] = float(energy_dist.atomic_weight_ratio)

    elif isinstance(energy_dist, MadlandNix):
        result["energy_dist_type"] = "madland-nix"
        result["energy_restriction_u"] = float(energy_dist.efl)
        result["energy_threshold"] = float(energy_dist.efh)
        x, y, _, _ = _serialize_tabulated1d(energy_dist.tm)
        result["energy_param_x"] = x.tolist()
        result["energy_param_y"] = y.tolist()

    else:
        raise TypeError(f"Unsupported energy distribution type: {type(energy_dist)}")

    return result


def _serialize_correlated(dist):
    """Serialize a CorrelatedAngleEnergy distribution."""
    from endf.univariate import Tabular, Discrete, Mixture

    result = {}
    result["corr_energies"] = np.asarray(dist.energy, dtype=np.float64).tolist()
    result["corr_breakpoints"] = list(int(b) for b in dist.breakpoints)
    result["corr_interpolation"] = list(int(i) for i in dist.interpolation)

    mu_tabular = []
    for i, mu_i in enumerate(dist.mu):
        row = []
        for mu_ij in mu_i:
            if isinstance(mu_ij, (Tabular, Discrete)):
                row.append(mu_ij)
            else:
                row.append(mu_ij.to_tabular())
        mu_tabular.append(row)

    n_eout = sum(len(d) for d in dist.energy_out)
    eout = np.empty((5, n_eout), dtype=np.float64)

    n_mu = sum(sum(len(mu_ij.x) for mu_ij in mu_i) for mu_i in mu_tabular)
    mu = np.empty((3, n_mu), dtype=np.float64)

    offsets = np.empty(len(dist.energy_out), dtype=np.int32)
    interp = np.empty(len(dist.energy_out), dtype=np.int32)
    n_discrete = np.empty(len(dist.energy_out), dtype=np.int32)

    offset_e = 0
    offset_mu = 0
    for i, d in enumerate(dist.energy_out):
        n = len(d)
        offsets[i] = offset_e

        if isinstance(d, Mixture):
            discrete, continuous = d.distribution
            m = len(discrete)
            n_discrete[i] = m
            interp[i] = 1 if continuous.interpolation == 'histogram' else 2
            eout[0, offset_e:offset_e+m] = discrete.x
            eout[1, offset_e:offset_e+m] = discrete.p
            eout[2, offset_e:offset_e+m] = discrete.c
            eout[0, offset_e+m:offset_e+n] = continuous.x
            eout[1, offset_e+m:offset_e+n] = continuous.p
            eout[2, offset_e+m:offset_e+n] = continuous.c
        else:
            if isinstance(d, Tabular):
                n_discrete[i] = 0
                interp[i] = 1 if d.interpolation == 'histogram' else 2
            elif isinstance(d, Discrete):
                n_discrete[i] = n
                interp[i] = 1
            eout[0, offset_e:offset_e+n] = d.x
            eout[1, offset_e:offset_e+n] = d.p
            eout[2, offset_e:offset_e+n] = d.c

        for j_idx, mu_ij in enumerate(mu_tabular[i]):
            if isinstance(mu_ij, Discrete):
                eout[3, offset_e+j_idx] = 0
            else:
                eout[3, offset_e+j_idx] = 1 if mu_ij.interpolation == 'histogram' else 2
            eout[4, offset_e+j_idx] = offset_mu

            n_mu_pts = len(mu_ij.x)
            mu[0, offset_mu:offset_mu+n_mu_pts] = mu_ij.x
            mu[1, offset_mu:offset_mu+n_mu_pts] = mu_ij.p
            mu[2, offset_mu:offset_mu+n_mu_pts] = mu_ij.c
            offset_mu += n_mu_pts

        offset_e += n

    result["corr_eout_data"] = eout.ravel(order='C').tolist()
    result["corr_eout_offsets"] = offsets.tolist()
    result["corr_eout_interp"] = interp.tolist()
    result["corr_eout_n_discrete"] = n_discrete.tolist()
    result["corr_mu_data"] = mu.ravel(order='C').tolist()
    result["corr_mu_offsets"] = [int(eout[4, k]) for k in range(n_eout)]
    result["corr_mu_interp"] = [int(eout[3, k]) for k in range(n_eout)]
    return result


def _serialize_kalbach_mann(dist):
    """Serialize a KalbachMann distribution."""
    from endf.univariate import Tabular, Discrete, Mixture

    result = {}
    result["km_energies"] = np.asarray(dist.energy, dtype=np.float64).tolist()
    result["km_breakpoints"] = list(int(b) for b in dist.breakpoints)
    result["km_interpolation"] = list(int(i) for i in dist.interpolation)

    n_tuple = sum(len(d) for d in dist.energy_out)
    distribution = np.empty((5, n_tuple), dtype=np.float64)
    offsets = np.empty(len(dist.energy_out), dtype=np.int32)
    interp = np.empty(len(dist.energy_out), dtype=np.int32)
    n_discrete = np.empty(len(dist.energy_out), dtype=np.int32)
    j = 0

    for i, (eout, km_r, km_a) in enumerate(zip(
            dist.energy_out, dist.precompound, dist.slope)):
        n = len(eout)
        offsets[i] = j

        if isinstance(eout, Mixture):
            discrete, continuous = eout.distribution
            m = len(discrete)
            n_discrete[i] = m
            interp[i] = 1 if continuous.interpolation == 'histogram' else 2
            distribution[0, j:j+m] = discrete.x
            distribution[1, j:j+m] = discrete.p
            distribution[2, j:j+m] = discrete.c
            distribution[0, j+m:j+n] = continuous.x
            distribution[1, j+m:j+n] = continuous.p
            distribution[2, j+m:j+n] = continuous.c
        else:
            if isinstance(eout, Tabular):
                n_discrete[i] = 0
                interp[i] = 1 if eout.interpolation == 'histogram' else 2
            elif isinstance(eout, Discrete):
                n_discrete[i] = n
                interp[i] = 1
            distribution[0, j:j+n] = eout.x
            distribution[1, j:j+n] = eout.p
            distribution[2, j:j+n] = eout.c

        distribution[3, j:j+n] = km_r.y
        distribution[4, j:j+n] = km_a.y
        j += n

    result["km_data"] = distribution.ravel(order='C').tolist()
    result["km_offsets"] = offsets.tolist()
    result["km_interp"] = interp.tolist()
    result["km_n_discrete"] = n_discrete.tolist()
    return result


def _build_distribution_row(mt, product_idx, dist_idx, dist, applicability_obj):
    """Build a single distribution row dict."""
    from endf.angle_energy import (
        UncorrelatedAngleEnergy, CorrelatedAngleEnergy, KalbachMann,
        NBodyPhaseSpace,
    )

    row = {
        "reaction_mt": mt,
        "product_idx": product_idx,
        "dist_idx": dist_idx,
    }

    nullable_cols = [
        "applicability_data", "applicability_shape",
        "applicability_breakpoints", "applicability_interpolation",
        "angle_energies", "angle_mu_data", "angle_mu_offsets",
        "angle_mu_interpolation",
        "energy_dist_type", "energy_dist_energies", "energy_dist_interpolation",
        "energy_dist_data", "energy_dist_offsets", "energy_dist_out_interp",
        "energy_dist_n_discrete",
        "energy_param_x", "energy_param_y",
        "energy_param2_x", "energy_param2_y",
        "energy_restriction_u", "energy_threshold", "energy_mass_ratio",
        "energy_primary_flag", "energy_atomic_weight_ratio",
        "energy_discrete_energy",
        "corr_energies", "corr_breakpoints", "corr_interpolation",
        "corr_eout_data", "corr_eout_offsets", "corr_eout_interp",
        "corr_eout_n_discrete",
        "corr_mu_data", "corr_mu_offsets", "corr_mu_interp",
        "km_energies", "km_breakpoints", "km_interpolation",
        "km_data", "km_offsets", "km_interp", "km_n_discrete",
        "nbody_n", "nbody_total_mass", "nbody_atomic_weight_ratio",
        "nbody_q_value",
    ]
    for col in nullable_cols:
        row[col] = None

    if applicability_obj is not None:
        data, shape, bp, interp = _serialize_applicability(applicability_obj)
        row["applicability_data"] = data
        row["applicability_shape"] = shape
        row["applicability_breakpoints"] = bp
        row["applicability_interpolation"] = interp

    if isinstance(dist, UncorrelatedAngleEnergy):
        row["type"] = "uncorrelated"
        if dist.angle is not None:
            energies, mu_data, offsets, interp = _serialize_angle_distribution(dist.angle)
            row["angle_energies"] = energies
            row["angle_mu_data"] = mu_data
            row["angle_mu_offsets"] = offsets
            row["angle_mu_interpolation"] = interp
        if dist.energy is not None:
            energy_cols = _serialize_energy_distribution(dist.energy)
            row.update(energy_cols)

    elif isinstance(dist, CorrelatedAngleEnergy):
        row["type"] = "correlated"
        corr_cols = _serialize_correlated(dist)
        row.update(corr_cols)

    elif isinstance(dist, KalbachMann):
        row["type"] = "kalbach-mann"
        km_cols = _serialize_kalbach_mann(dist)
        row.update(km_cols)

    elif isinstance(dist, NBodyPhaseSpace):
        row["type"] = "nbody"
        row["nbody_n"] = int(dist.n_particles)
        row["nbody_total_mass"] = float(dist.total_mass)
        row["nbody_atomic_weight_ratio"] = float(dist.atomic_weight_ratio)
        row["nbody_q_value"] = float(dist.q_value)

    else:
        raise TypeError(f"Unsupported distribution type: {type(dist)}")

    return row


# Option-D hosting format: LZ4-compressed Arrow IPC (decoded by arrow-rs's
# pure-Rust `lz4` feature, so the yamc download build stays C-free).
_LZ4 = ipc.IpcWriteOptions(compression="lz4")


def _write_arrow_ipc(table, filepath):
    """Write a PyArrow table as Arrow IPC (option-D: LZ4-compressed)."""
    with pa.OSFile(str(filepath), 'wb') as f:
        writer = ipc.new_file(f, table.schema, options=_LZ4)
        writer.write_table(table)
        writer.close()


def _write_arrow_ipc_per_row(table, filepath):
    """Write one record batch per row (option-D reactions.arrow), so the webapp
    can range-read a single MT without pulling the whole file. LZ4-compressed;
    the yamc loader fuses the batches back into one table on read."""
    with pa.OSFile(str(filepath), 'wb') as f:
        writer = ipc.new_file(f, table.schema, options=_LZ4)
        for i in range(table.num_rows):
            writer.write_batch(table.slice(i, 1).combine_chunks().to_batches()[0])
        writer.close()


def export_neutron_to_arrow(data, path, *, library="", data_version=""):
    """Export an IncidentNeutron object to a simulation-ready .arrow/ directory.

    Parameters
    ----------
    data : endf.IncidentNeutron
        The incident neutron data to export.
    path : str or Path
        Directory path to write Arrow files to (e.g., "Li6.arrow").
    library : str, optional
        Library name (e.g., "endfb-8.0", "fendl-3.2c").
    data_version : str, optional
        Identifier of the published release this directory belongs to, stamped
        into ``version.json``. See :func:`completion.write_completion_marker`.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # version.json is written at the end, not here: it is the completion
    # marker the resume path skips on, so it has to mean every table below
    # made it to disk. See completion.py.

    # ------------------------------------------------------------------
    # 1. nuclide.arrow
    # ------------------------------------------------------------------
    temperatures = list(data.temperatures)
    kts = [float(data.kTs[i]) for i in range(len(temperatures))]

    energy_temps = list(data.energy.keys())
    energy_vals = [np.asarray(data.energy[t], dtype=np.float64).tolist()
                   for t in energy_temps]

    nuclide_table = pa.table(
        {
            "name": [data.name],
            "Z": [int(data.atomic_number)],
            "A": [int(data.mass_number)],
            "atomic_weight_ratio": [float(data.atomic_weight_ratio)],
            "temperatures": [temperatures],
            "kTs": [kts],
            "energy_temperatures": [energy_temps],
            "energy_values": [energy_vals],
        },
        schema=NUCLIDE_SCHEMA,
    )
    _write_arrow_ipc(nuclide_table, path / "nuclide.arrow")

    # ------------------------------------------------------------------
    # 2. Synthesize MTs + build fast_xs for each temperature
    # ------------------------------------------------------------------
    all_synthetic = {}  # temperature -> {mt: xs_array}
    fast_xs_rows = []

    for temp in temperatures:
        synth = synthesize_hierarchical_mts(data, temp)
        all_synthetic[temp] = synth
        fxs = build_fast_xs(data, temp, synth)

        xs_shape = list(fxs["xs"].shape)
        scatter_shape = list(fxs["scatter_mt_xs"].shape)
        fission_shape = list(fxs["fission_mt_xs"].shape)

        fast_xs_rows.append({
            "temperature": temp,
            "log_e_min": fxs["log_e_min"],
            "inv_log_delta": fxs["inv_log_delta"],
            "log_grid_index": fxs["log_grid_index"].tolist(),
            "xs": fxs["xs"].ravel(order='C').tolist(),
            "xs_shape": xs_shape,
            "energy": fxs["energy"].tolist(),
            "scatter_mt_numbers": fxs["scatter_mt_numbers"],
            "scatter_mt_xs": fxs["scatter_mt_xs"].ravel(order='C').tolist(),
            "scatter_mt_shape": scatter_shape,
            "fission_mt_numbers": fxs["fission_mt_numbers"],
            "fission_mt_xs": fxs["fission_mt_xs"].ravel(order='C').tolist(),
            "fission_mt_shape": fission_shape,
            "has_partial_fission": fxs["has_partial_fission"],
            "xs_ngamma": fxs["xs_ngamma"].tolist(),
            "photon_prod": fxs["photon_prod"].tolist(),
        })

    # Write fast_xs.arrow
    if fast_xs_rows:
        fast_xs_table = pa.table(
            {col: [r[col] for r in fast_xs_rows]
             for col in FAST_XS_SCHEMA.names},
            schema=FAST_XS_SCHEMA,
        )
        _write_arrow_ipc(fast_xs_table, path / "fast_xs.arrow")

    # ------------------------------------------------------------------
    # 3. reactions.arrow + products.arrow + distributions.arrow
    # ------------------------------------------------------------------
    reaction_rows = []
    product_rows = []
    distribution_rows = []
    total_nu_written = False
    total_nu_row = None

    # First, write original (non-synthetic) reactions
    for rx in data.reactions.values():
        label = REACTION_NAME.get(rx.MT, str(rx.MT))

        xs_temps = []
        xs_vals = []
        xs_thresh = []
        for T in rx.xs:
            xs_temps.append(T)
            if rx.xs[T] is not None:
                xs_vals.append(np.asarray(rx.xs[T].y, dtype=np.float64).tolist())
                threshold_idx = getattr(rx.xs[T], '_threshold_idx', 0)
                xs_thresh.append(int(threshold_idx))
            else:
                xs_vals.append([])
                xs_thresh.append(0)

        reaction_rows.append({
            "mt": int(rx.MT),
            "label": label,
            "Q_value": float(rx.q_reaction),
            "center_of_mass": bool(rx.center_of_mass),
            "redundant": bool(rx.redundant),
            "xs_temperatures": xs_temps,
            "xs_values": xs_vals,
            "xs_threshold_idx": xs_thresh,
        })

        # Products
        for p_idx, p in enumerate(rx.products):
            yield_type, yield_data, yield_shape, yield_bp, yield_interp = \
                _serialize_yield(p.yield_)

            product_rows.append({
                "reaction_mt": int(rx.MT),
                "product_idx": p_idx,
                "particle": str(p.name),
                "emission_mode": str(p.emission_mode),
                "decay_rate": float(p.decay_rate),
                "n_distribution": len(p.distribution),
                "yield_type": yield_type,
                "yield_data": yield_data,
                "yield_shape": yield_shape,
                "yield_breakpoints": yield_bp,
                "yield_interpolation": yield_interp,
            })

            # Distributions
            for d_idx, d in enumerate(p.distribution):
                applicability_obj = None
                if p.applicability and d_idx < len(p.applicability):
                    applicability_obj = p.applicability[d_idx]
                dist_row = _build_distribution_row(
                    rx.MT, p_idx, d_idx, d, applicability_obj)
                distribution_rows.append(dist_row)

        # total_nu
        if not total_nu_written and len(rx.derived_products) > 0:
            dp = rx.derived_products[0]
            total_nu_written = True
            yield_type, yield_data, yield_shape, yield_bp, yield_interp = \
                _serialize_yield(dp.yield_)
            total_nu_row = {
                "particle": str(dp.name),
                "emission_mode": str(dp.emission_mode),
                "decay_rate": float(dp.decay_rate),
                "yield_type": yield_type,
                "yield_data": yield_data,
                "yield_shape": yield_shape,
                "yield_breakpoints": yield_bp,
                "yield_interpolation": yield_interp,
            }

    # Now add synthesized MT rows.
    #
    # Labelled with REACTION_NAME, the same as every other row. There used to be
    # an override here giving MT 3, 4 and 101 their own spellings, "(n,non-elastic)",
    # "(n,inelastic)" and "(n,disappearance)", none of which is what the format
    # calls those reactions. The result was two vocabularies in one file, only
    # one of which resolved: a score copied out of reactions.arrow was rejected
    # by yamc (issue #438). MT 1 and 27 were in that table too, spelled exactly
    # as REACTION_NAME already spells them, which is how it went unnoticed.
    for mt in sorted(SYNTHETIC_MTS):
        # Skip if this MT already exists as an original reaction
        if mt in data.reactions:
            continue

        xs_temps = []
        xs_vals = []
        xs_thresh = []
        for temp in temperatures:
            xs_temps.append(temp)
            synth_xs = all_synthetic[temp].get(mt)
            if synth_xs is not None:
                xs_vals.append(synth_xs.tolist())
            else:
                xs_vals.append([])
            xs_thresh.append(0)

        reaction_rows.append({
            "mt": mt,
            "label": REACTION_NAME.get(mt, str(mt)),
            "Q_value": 0.0,
            "center_of_mass": False,
            "redundant": True,
            "xs_temperatures": xs_temps,
            "xs_values": xs_vals,
            "xs_threshold_idx": xs_thresh,
        })

    # Write reactions
    if reaction_rows:
        reactions_table = pa.table(
            {col: [r[col] for r in reaction_rows]
             for col in REACTIONS_SCHEMA.names},
            schema=REACTIONS_SCHEMA,
        )
        _write_arrow_ipc_per_row(reactions_table, path / "reactions.arrow")

    # Write products
    if product_rows:
        products_table = pa.table(
            {col: [r[col] for r in product_rows]
             for col in PRODUCTS_SCHEMA.names},
            schema=PRODUCTS_SCHEMA,
        )
        _write_arrow_ipc(products_table, path / "products.arrow")

    # Write distributions
    if distribution_rows:
        distributions_table = pa.table(
            {col: [r[col] for r in distribution_rows]
             for col in DISTRIBUTIONS_SCHEMA.names},
            schema=DISTRIBUTIONS_SCHEMA,
        )
        _write_arrow_ipc(distributions_table, path / "distributions.arrow")

    # ------------------------------------------------------------------
    # 4. urr.arrow (optional)
    # ------------------------------------------------------------------
    if data.urr:
        urr_rows = []
        for temperature, urr in data.urr.items():
            table_arr = np.asarray(urr.table, dtype=np.float64)
            urr_rows.append({
                "temperature": temperature,
                "energy": np.asarray(urr.energy, dtype=np.float64).tolist(),
                "table_data": table_arr.ravel(order='C').tolist(),
                "table_shape": list(table_arr.shape),
                "interpolation": int(urr.interpolation),
                "inelastic": int(urr.inelastic_flag),
                "absorption": int(urr.absorption_flag),
                "multiply_smooth": bool(urr.multiply_smooth),
            })
        urr_table = pa.table(
            {col: [r[col] for r in urr_rows] for col in URR_SCHEMA.names},
            schema=URR_SCHEMA,
        )
        _write_arrow_ipc(urr_table, path / "urr.arrow")

    # ------------------------------------------------------------------
    # 4b. fission_photon.arrow (optional) -- yamc issue #369
    # ------------------------------------------------------------------
    # A fission releases photons from the fission products' decay as well as
    # promptly, and the evaluation's prompt photon production does not include
    # them. OpenMC scales fission photon production by
    # f(E) = (prompt + delayed) / prompt (settings::delayed_photon_scaling, on by
    # default); without these terms yamc cannot, and actinide fission photon
    # production comes out ~38% low.
    #
    # Present for 79 of 557 ENDF/B-VIII.1 nuclides. Note this is populated by
    # IncidentNeutron.from_endf / from_njoy, which read MF=1/MT=458 from the
    # original evaluation, but NOT by from_ace -- an ACE-sourced conversion
    # silently has no fission energy release data to write.
    fission_photon_rows = _fission_photon_rows(data)
    if fission_photon_rows:
        fission_photon_table = pa.table(
            {col: [r[col] for r in fission_photon_rows]
             for col in FISSION_PHOTON_SCHEMA.names},
            schema=FISSION_PHOTON_SCHEMA,
        )
        _write_arrow_ipc(fission_photon_table, path / "fission_photon.arrow")

    # ------------------------------------------------------------------
    # 5. total_nu.arrow (optional)
    # ------------------------------------------------------------------
    if total_nu_written and total_nu_row is not None:
        total_nu_table = pa.table(
            {col: [total_nu_row[col]] for col in TOTAL_NU_SCHEMA.names},
            schema=TOTAL_NU_SCHEMA,
        )
        _write_arrow_ipc(total_nu_table, path / "total_nu.arrow")

    # ------------------------------------------------------------------
    # 6. version.json, last, so it means the whole directory is there
    # ------------------------------------------------------------------
    write_completion_marker(path, library, data_version)
