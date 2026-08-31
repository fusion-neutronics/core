"""Arrow schema definitions for simulation-ready nuclear data tables.

The declarations live in Rust, in `crates/nuclear-data-schema`, so that the
readers and writers on that side build against them and a rename is a compile
error. This module reconstructs the same pyarrow schemas from the rendered
manifest that crate emits, which keeps this package pure Python with no
compiled dependency: the transmutation half has to stay installable without
yamc (#381).

Regenerate the manifest after changing the Rust declarations:

    cargo run -p nuclear-data-schema --bin emit-schema-manifest \
        > packages/nuclear_data_to_arrow/src/nuclear_data_to_arrow/schema/manifest.json

`test_schema_manifest.py` fails if the committed manifest and the crate have
drifted apart.

The manifest sits inside the package rather than beside it so that it ships in
the wheel. Resolved from anywhere outside, an installed copy would point at
`site-packages/../../schema/manifest.json`, which is nothing.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

import pyarrow as pa

MANIFEST_PATH = Path(__file__).resolve().parent / "schema" / "manifest.json"

_PRIMITIVES = {
    "string": pa.utf8(),
    "double": pa.float64(),
    "int32": pa.int32(),
    "bool": pa.bool_(),
}


def _arrow_type(rendered):
    """Rebuild a pyarrow type from the manifest's stable spelling."""
    inner = re.fullmatch(r"list<(.+)>", rendered)
    if inner:
        return pa.list_(_arrow_type(inner.group(1)))
    try:
        return _PRIMITIVES[rendered]
    except KeyError:
        raise ValueError(f"unknown Arrow type in the manifest: {rendered!r}") from None


@lru_cache(maxsize=1)
def _manifest():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} is missing. Regenerate it with:\n"
            "  cargo run -p nuclear-data-schema --bin emit-schema-manifest "
            f"> {MANIFEST_PATH}"
        )
    return json.loads(MANIFEST_PATH.read_text())


def _schema(section):
    body = _manifest()["sections"][section]
    fields = [
        pa.field(f["name"], _arrow_type(f["type"]), nullable=f["nullable"])
        for f in body["fields"]
    ]
    metadata = {k.encode(): v.encode() for k, v in body["metadata"].items()} or None
    return pa.schema(fields, metadata=metadata)


def schema_manifest():
    """The on-disk format as plain JSON-serialisable data, as emitted by Rust."""
    return _manifest()


NUCLIDE_SCHEMA = _schema("nuclide.arrow")
REACTIONS_SCHEMA = _schema("reactions.arrow")
PRODUCTS_SCHEMA = _schema("products.arrow")
DISTRIBUTIONS_SCHEMA = _schema("distributions.arrow")
URR_SCHEMA = _schema("urr.arrow")
FISSION_PHOTON_SCHEMA = _schema("fission_photon.arrow")
TOTAL_NU_SCHEMA = _schema("total_nu.arrow")
FAST_XS_SCHEMA = _schema("fast_xs.arrow")
COVARIANCE_SCHEMA = _schema("covariance.arrow")
ELEMENT_SCHEMA = _schema("element.arrow")
SUBSHELLS_SCHEMA = _schema("subshells.arrow")
COMPTON_SCHEMA = _schema("compton.arrow")
BREMSSTRAHLUNG_SCHEMA = _schema("bremsstrahlung.arrow")
DECAY_NUCLIDES_SCHEMA = _schema("decay/nuclides.arrow")
DECAY_MODES_SCHEMA = _schema("decay/decay_modes.arrow")
DECAY_SOURCES_SCHEMA = _schema("decay/sources.arrow")
TRANSMUTATION_REACTIONS_SCHEMA = _schema("reactions/reactions.arrow")
FISSION_YIELDS_SCHEMA = _schema("fission_yields/fission_yields.arrow")
FISSION_YIELD_ALIASES_SCHEMA = _schema("fission_yields/aliases.arrow")
BRANCHING_SCHEMA = _schema("branching/branching.arrow")


SECTION_SCHEMAS = {
    # neutron, {nuclide}.arrow/
    "nuclide.arrow": NUCLIDE_SCHEMA,
    "reactions.arrow": REACTIONS_SCHEMA,
    "products.arrow": PRODUCTS_SCHEMA,
    "distributions.arrow": DISTRIBUTIONS_SCHEMA,
    "fast_xs.arrow": FAST_XS_SCHEMA,
    "urr.arrow": URR_SCHEMA,
    "total_nu.arrow": TOTAL_NU_SCHEMA,
    "fission_photon.arrow": FISSION_PHOTON_SCHEMA,
    # Declared, but written only by yamc-convert: the covariance path is
    # Rust-only (#514). The schema still belongs here, since this mirrors
    # what the format declares rather than what this package writes.
    "covariance.arrow": COVARIANCE_SCHEMA,
    # photon, {element}.arrow/
    "element.arrow": ELEMENT_SCHEMA,
    "subshells.arrow": SUBSHELLS_SCHEMA,
    "compton.arrow": COMPTON_SCHEMA,
    "bremsstrahlung.arrow": BREMSSTRAHLUNG_SCHEMA,
    # transmutation, transmutation_{library}.arrow/<subsection>/
    "decay/nuclides.arrow": DECAY_NUCLIDES_SCHEMA,
    "decay/decay_modes.arrow": DECAY_MODES_SCHEMA,
    "decay/sources.arrow": DECAY_SOURCES_SCHEMA,
    "reactions/reactions.arrow": TRANSMUTATION_REACTIONS_SCHEMA,
    "fission_yields/fission_yields.arrow": FISSION_YIELDS_SCHEMA,
    "fission_yields/aliases.arrow": FISSION_YIELD_ALIASES_SCHEMA,
    "branching/branching.arrow": BRANCHING_SCHEMA,
}
