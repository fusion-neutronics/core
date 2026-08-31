"""Rebuilding pyarrow schemas from the manifest.

crates/nuclear-data-schema declares the format and emits the manifest; this
package reconstructs pyarrow schemas from it. Only that reconstruction is
Python's responsibility, so it is the only thing tested here.

Currency of the manifest is not: comparing it against schemas built from it is
a tautology, since both sides move together. That check lives in
crates/nuclear-data-schema/tests/manifest_is_current.rs, which regenerates the
manifest from the declarations and diffs it.

There used to be a test here that scanned the Rust readers for
`get_f64(batch, "col")` accessors and checked those names against the manifest.
It is gone: it compared against the union of all sections, because an accessor
does not say which file it reads, so it could not catch a column read from the
wrong section, and it silently covered nothing at all for `chain_arrow.rs`,
which uses a different accessor style. `nuclear_data_schema::check_batch`
replaces it, per section, at the point of reading.
"""

import json
from pathlib import Path

import pyarrow as pa
import pytest

from nuclear_data_to_arrow import schemas
from nuclear_data_to_arrow.schemas import (
    MANIFEST_PATH,
    SECTION_SCHEMAS,
    _arrow_type,
    schema_manifest,
)


def test_manifest_is_where_the_package_expects_it():
    assert MANIFEST_PATH.is_file(), f"missing {MANIFEST_PATH}"
    # Inside the importable package, not beside it. The module cannot import
    # without the manifest, so anywhere else and the wheel is dead on arrival:
    # only files under the package directory are picked up by package-data, and
    # an editable install hides the difference by reading the source tree.
    assert MANIFEST_PATH == Path(schemas.__file__).resolve().parent / "schema" / "manifest.json"
    doc = json.loads(MANIFEST_PATH.read_text())
    assert doc["format_version"] == 1
    assert doc["sections"], "manifest declares no sections"


@pytest.mark.parametrize(
    "rendered, expected",
    [
        ("string", pa.utf8()),
        ("double", pa.float64()),
        ("int32", pa.int32()),
        ("bool", pa.bool_()),
        ("list<double>", pa.list_(pa.float64())),
        ("list<int32>", pa.list_(pa.int32())),
        ("list<string>", pa.list_(pa.utf8())),
        ("list<list<double>>", pa.list_(pa.list_(pa.float64()))),
    ],
)
def test_every_rendered_type_reconstructs(rendered, expected):
    """The eight spellings the emitter produces must all round-trip.

    A spelling this cannot parse would otherwise surface as a confusing
    KeyError deep in a writer.
    """
    assert _arrow_type(rendered) == expected


def test_an_unknown_type_is_refused_clearly():
    with pytest.raises(ValueError, match="unknown Arrow type"):
        _arrow_type("decimal128")


def test_reconstructed_schemas_have_the_declared_shape():
    """Spot-check the reconstruction against known fields.

    Hand-written rather than derived from the manifest, so this fails if the
    reconstruction silently changes what it builds.
    """
    nuclide = SECTION_SCHEMAS["nuclide.arrow"]
    assert nuclide.field("name").type == pa.utf8()
    assert nuclide.field("Z").type == pa.int32()
    assert nuclide.field("atomic_weight_ratio").type == pa.float64()
    assert nuclide.field("energy_values").type == pa.list_(pa.list_(pa.float64()))
    assert nuclide.metadata == {b"filetype": b"data_neutron", b"version": b"4.0"}

    branching = SECTION_SCHEMAS["branching/branching.arrow"]
    assert not branching.field("nuclide").nullable
    assert branching.field("energy").type == pa.list_(pa.float64())


def test_every_section_reconstructs_into_a_usable_schema():
    manifest = schema_manifest()
    assert set(manifest["sections"]) == set(SECTION_SCHEMAS)
    for name, schema in SECTION_SCHEMAS.items():
        assert isinstance(schema, pa.Schema), name
        assert len(schema) == len(manifest["sections"][name]["fields"]), name
        # A schema with no fields would let a writer produce an empty table.
        assert len(schema) > 0, name
