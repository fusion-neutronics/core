"""Stable digests of an ``.arrow/`` output tree, for regression testing.

Committing whole Arrow trees as golden files would cost megabytes, and hashing
the raw bytes would be fragile because the IPC framing and LZ4 block layout can
change with the pyarrow version. So this hashes the *decoded* value of every
column instead: the same thing a column-by-column diff compares, in a few
kilobytes, and stable across pyarrow releases.

Floats are quantised to ``SIGNIFICANT_DIGITS`` before hashing, because a
bit-exact digest is not reproducible across machines. Anything computed through
``log``, ``exp`` or an interpolating spline can differ in the last bit or two
between platforms, libm versions and scipy releases, which in the photon output
covers the log-space cross sections, the bremsstrahlung electron energy grid and
its differential cross sections. Twelve digits leaves four orders of magnitude
above that noise and five below the smallest real defect this guard is meant to
catch, a 2.6e-07 relative error in a synthesised total.

The quantisation is a genuine trade: a value sitting exactly on a rounding
boundary could still flip and produce a spurious failure. Twelve digits makes
that unlikely rather than impossible, and the alternative, committing full
precision values, costs megabytes and reintroduces the cross-platform problem.

``digest_tree`` produces a mapping that can be written to JSON and compared, and
``compare`` reports the differences in a form that says which column moved.
"""

import hashlib
import json
from pathlib import Path

import pyarrow as pa

# Provenance fields that legitimately differ between runs.
VOLATILE_JSON_KEYS = {"created_utc", "converter_version"}

# Significant digits retained before hashing a float. See the module docstring.
SIGNIFICANT_DIGITS = 12


def _canonical(value):
    """Render a decoded Arrow value deterministically, quantising floats."""
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return repr(value)
        # Normalise the zeros so -0.0 and 0.0 agree.
        if value == 0.0:
            return "0"
        return f"%.{SIGNIFICANT_DIGITS - 1}e" % value
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{_canonical(v)}"
                              for k, v in sorted(value.items())) + "}"
    return repr(value)


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def digest_tree(path):
    """Return ``{relative path: {column or key: digest}}`` for an output tree.

    Arrow tables contribute one digest per column plus the row count and the
    schema metadata. JSON files contribute one digest over their contents with
    the volatile provenance keys removed.
    """
    path = Path(path)
    out = {}
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = file.relative_to(path).as_posix()
        if file.suffix == ".arrow":
            with pa.memory_map(str(file), "r") as source:
                table = pa.ipc.open_file(source).read_all()
            entry = {
                "__rows__": table.num_rows,
                "__schema__": _hash(_canonical(
                    [(f.name, str(f.type)) for f in table.schema])),
            }
            if table.schema.metadata:
                entry["__metadata__"] = _hash(_canonical(
                    {k.decode(): v.decode()
                     for k, v in table.schema.metadata.items()}))
            for name in table.schema.names:
                entry[name] = _hash(_canonical(table.column(name).to_pylist()))
            out[rel] = entry
        elif file.suffix == ".json":
            payload = {k: v for k, v in json.loads(file.read_text()).items()
                       if k not in VOLATILE_JSON_KEYS}
            out[rel] = {"__json__": _hash(_canonical(payload))}
        else:
            out[rel] = {"__bytes__": _hash(_canonical(file.read_bytes()))}
    return out


def compare(expected, actual):
    """Return a list of human-readable differences between two digest trees."""
    diffs = []
    for rel in sorted(set(expected) - set(actual)):
        diffs.append(f"missing file: {rel}")
    for rel in sorted(set(actual) - set(expected)):
        diffs.append(f"unexpected file: {rel}")
    for rel in sorted(set(expected) & set(actual)):
        exp, act = expected[rel], actual[rel]
        for key in sorted(set(exp) - set(act)):
            diffs.append(f"{rel}: missing column {key}")
        for key in sorted(set(act) - set(exp)):
            diffs.append(f"{rel}: unexpected column {key}")
        for key in sorted(set(exp) & set(act)):
            if exp[key] != act[key]:
                diffs.append(f"{rel}: {key} changed "
                             f"({exp[key]} -> {act[key]})")
    return diffs


def write(path, digests):
    Path(path).write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")


def read(path):
    return json.loads(Path(path).read_text())
