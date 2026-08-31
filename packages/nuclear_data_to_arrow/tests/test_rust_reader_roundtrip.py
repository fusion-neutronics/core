"""Read what this package writes back with yamc's Rust reader, not a Python one.

The point of the round trip is to cross the language boundary. The tests in
test_neutron_roundtrip.py and test_photon_roundtrip.py read the output with
this package's own Python readers, so writer and reader can be wrong together
in exactly the same way, and the suite stays green.

That is not hypothetical. Issue #379: the transmutation writer spelled MT 18
"fission" (the chain-file spelling), while the Rust consumer's reaction map
only held "(n,fission)". Nothing matched, no fission product was ever produced
by transmutation, and every Python-side test passed. A vocabulary mismatch is
invisible to any reader that shares the writer's vocabulary.

These tests are skipped when yamc is not importable, so the converter's own CI
(which builds no Rust) stays green; they run in the monorepo, which is the
reason the package lives here.
"""

from types import SimpleNamespace

import pytest

from nuclear_data_to_arrow import (
    convert_neutron,
    convert_photon,
    export_transmutation_to_arrow,
)

yamc = pytest.importorskip("yamc", reason="needs the compiled yamc extension")


# --- neutron: writer -> Rust reader ----------------------------------------

def test_neutron_arrow_loads_in_the_rust_reader(li6_ace_path, tmp_path):
    """A directory this package writes must satisfy yamc's own loader.

    Covers the schema contract as a whole: field names, types, nullability and
    the format_version gate in nuclide_arrow.rs. A rename on either side fails
    here rather than at someone's simulation start.
    """
    arrow_path = convert_neutron(li6_ace_path, tmp_path,
                                 source_format="ace", library="endfb-8.1")

    nuclide = yamc.Nuclide("Li6")
    nuclide.read_nuclear_data(str(arrow_path))

    assert nuclide.name == "Li6"
    assert nuclide.atomic_weight_ratio > 0
    assert nuclide.reactions, "the Rust reader loaded no reactions"


def test_format_version_gate_is_live(li6_ace_path, tmp_path):
    """The reader really does police format_version, so bumping it is safe.

    Three version numbers are stamped into the output and only this one is
    checked: schemas.py puts "4.0" in the Arrow schema metadata,
    transmutation_writer.py stamps FORMAT_VERSION 2 for transmutation, and
    completion.py stamps format_version 1 here. This test pins the one that
    has teeth, so a writer-side bump fails loudly against an old reader rather
    than being parsed as though nothing changed.
    """
    import json

    arrow_path = convert_neutron(li6_ace_path, tmp_path,
                                 source_format="ace", library="endfb-8.1")
    marker = arrow_path / "version.json"
    info = json.loads(marker.read_text())
    assert info["format_version"] == 1, "writer bumped it; update the reader too"

    info["format_version"] = 99
    marker.write_text(json.dumps(info))

    with pytest.raises(Exception, match="[Ff]ormat version"):
        yamc.Nuclide("Li6").read_nuclear_data(str(arrow_path))


def test_rust_reader_sees_the_same_mt_numbers(li6_ace_path, tmp_path):
    """The MTs the writer emitted are the MTs the reader materialises."""
    import pyarrow.ipc as ipc

    arrow_path = convert_neutron(li6_ace_path, tmp_path,
                                 source_format="ace", library="endfb-8.1")

    with open(arrow_path / "reactions.arrow", "rb") as f:
        written = {row["mt"] for row in ipc.open_file(f).read_all().to_pylist()}

    nuclide = yamc.Nuclide("Li6")
    nuclide.read_nuclear_data(str(arrow_path))
    read_back = {mt for mts in nuclide.reactions.values() for mt in mts}

    missing = written - read_back
    assert not missing, f"the Rust reader dropped MTs the writer emitted: {sorted(missing)}"


# --- the readback gate itself (#443) ---------------------------------------

def test_the_readback_gate_reports_the_scope_it_actually_loaded(li6_ace_path, tmp_path):
    """A full-scope read must come back full, not quietly narrowed.

    `read_nuclide_from_arrow` takes the path verbatim and reports what it got,
    which is what makes it a gate rather than a load. It matters because the
    loader does NOT fail on a directory holding no transport sections: it
    narrows a "full" request to cross-sections-only, on the theory that this is
    a cross-sections-only conversion. So a gate that asks for "full" and does
    not look at `scope_loaded` passes a directory with no distributions, no
    products and no fast_xs.
    """
    arrow_path = convert_neutron(li6_ace_path, tmp_path,
                                 source_format="ace", library="endfb-8.1")

    summary = yamc.read_nuclide_from_arrow(str(arrow_path))

    assert summary["scope_loaded"] == "full", (
        "the loader narrowed a full read, so this conversion is missing "
        "transport sections"
    )
    assert summary["name"] == "Li6"
    assert summary["atomic_weight_ratio"] > 0
    assert summary["mts"], "the Rust reader materialised no reactions"
    assert summary["energy_points"], "no per-temperature energy grid came back"


def test_photon_arrow_loads_in_the_rust_reader(photon_endf_paths, tmp_path):
    """The photon half crosses the language boundary too.

    There was no way to write this until `read_element_from_arrow` existed
    (#443): the photon writer's only reader was this package's own
    `read_photon_from_arrow`, which shares its vocabulary. The three auxiliary
    tabulations asserted below are the ones that are in no evaluation, so a
    conversion that drops them reads back perfectly well and is still not what
    transport wants.
    """
    photoat_path, atom_path = photon_endf_paths
    arrow_path = convert_photon(photoat_path, tmp_path, atom_path=atom_path,
                                library="endfb-8.1")

    summary = yamc.read_element_from_arrow(str(arrow_path))

    assert summary["name"] == "Fe"
    assert summary["atomic_number"] == 26
    assert summary["n_energy_points"] > 0
    assert summary["n_subshells"] > 0
    assert summary["has_atomic_relaxation"], "the relaxation ENDF was dropped"
    assert summary["has_compton_profiles"], "the Compton profiles were dropped"


# --- transmutation: the #379 guard -----------------------------------------

def _reaction(type_, target, Q=0.0, br=1.0):
    return SimpleNamespace(type=type_, target=target, Q=Q, branching_ratio=br)


def _decay_mode(type_, target, br=1.0):
    return SimpleNamespace(type=type_, target=target, branching_ratio=br)


def _fy_entry(products, yields):
    return SimpleNamespace(products=products, yields=yields)


def _fissioning_chain():
    """A minimal network whose only interesting feature is a fission channel.

    "fission" is the spelling the chain files use and therefore the spelling
    this package writes. If the reader's vocabulary disagrees, the reaction
    silently vanishes rather than erroring, which is precisely #379.
    """
    u235 = SimpleNamespace(
        name="U235", half_life=2.2e16, decay_energy=5.0e6,
        decay_modes=[_decay_mode("alpha", "Th231")],
        sources={},
        reactions=[
            _reaction("(n,gamma)", "U236", Q=6.5e6),
            _reaction("fission", None, Q=2.0e8),
        ],
        yield_data={0.0253: _fy_entry(["Xe135", "Cs137"], [0.06, 0.062])},
    )
    u236 = SimpleNamespace(
        name="U236", half_life=7.4e14, decay_energy=4.6e6,
        decay_modes=[_decay_mode("alpha", "Th232")],
        sources={}, reactions=[], yield_data=None,
    )
    inert = [
        SimpleNamespace(name=n, half_life=None, decay_energy=0.0,
                        decay_modes=[], sources={}, reactions=[], yield_data=None)
        for n in ("Th231", "Th232", "Xe135", "Cs137")
    ]
    return SimpleNamespace(nuclides=[u235, u236, *inert])


@pytest.fixture
def fissioning_chain_arrow(tmp_path):
    out = tmp_path / "transmutation_endfb-8.1.arrow"
    export_transmutation_to_arrow(_fissioning_chain(), out, library="endfb-8.1")
    return out


def test_transmutation_arrow_loads_in_the_rust_reader(fissioning_chain_arrow):
    chain = yamc.TransmutationChain(str(fissioning_chain_arrow))
    assert "U235" in chain.nuclide_names


def test_written_reactions_survive_the_parse(fissioning_chain_arrow):
    """Rows written are rows parsed.

    Note what this does NOT prove: the parser echoes reaction-type strings back
    verbatim, so it passes for a type string the transmutation engine cannot
    resolve. That semantic check is
    test_writer_vocabulary_is_a_subset_of_the_rust_reaction_map below.
    """
    import pyarrow.ipc as ipc

    path = fissioning_chain_arrow / "reactions" / "reactions.arrow"
    with open(path, "rb") as f:
        rows = ipc.open_file(f).read_all().to_pylist()
    written = {(r["nuclide"], r["type"]) for r in rows}
    assert ("U235", "fission") in written, "fixture no longer exercises fission"

    # chain.reactions is {nuclide: [(type, target, branching_ratio), ...]}.
    chain = yamc.TransmutationChain(str(fissioning_chain_arrow))
    read_back = {
        (nuclide, reaction_type)
        for nuclide, rs in chain.reactions.items()
        for reaction_type, _target, _br in rs
    }
    assert not written - read_back


# --- the real #379 guard: the two vocabularies must agree -------------------

def _rust_reaction_mt_map():
    """Parse REACTION_MT_MAP out of crates/yani/src/reactions.rs.

    Read from source rather than called, because reaction_type_to_mt is not
    exposed through the Python bindings. Reading the table is enough: it *is*
    the vocabulary, and a name absent from it resolves to no MT.

    The table moved here from yani-transmute when the duplicate copies were
    consolidated, which is how `(n,2nd)` came to be mapped to MT 35 in one of
    them. This test kept looking at the old path and started failing on the
    regex rather than on the vocabulary, so the path is asserted below: a
    silent skip here would put the #379 guard back to sleep.
    """
    import pathlib
    import re

    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        src = parent / "crates" / "yani" / "src" / "reactions.rs"
        if src.is_file():
            break
    else:
        pytest.skip("not in the monorepo checkout, so the Rust source is absent")

    body = re.search(r"REACTION_MT_MAP: &\[\(&str, i32\)\] = &\[(.*?)\n\];",
                     src.read_text(), re.S)
    assert body, "REACTION_MT_MAP is no longer shaped as this test expects"
    pairs = re.findall(r'\("([^"]+)",\s*(\d+)\)', body.group(1))
    assert pairs, "parsed no entries from REACTION_MT_MAP"
    return {name: int(mt) for name, mt in pairs}


def test_writer_vocabulary_is_a_subset_of_the_rust_reaction_map():
    """Every reaction name the writer can emit must resolve on the Rust side.

    This is the assertion whose absence was #379. The chain files spell MT 18
    "fission"; yani-transmute's map held only "(n,fission)". The name resolved
    to no MT, so no fission rate was ever computed, no fission product was ever
    produced, and nothing errored. Both sides' own tests passed, because
    neither could see the other's vocabulary.

    It is only writable here: it needs the Python writer and the Rust source in
    one checkout.
    """
    from endf.chain import REACTIONS

    rust = _rust_reaction_mt_map()

    # The writer emits every name in endf-python's REACTIONS table, plus
    # "fission" for MT 18, which REACTIONS itself does not carry (verified by
    # the assertion below, so this list cannot silently rot).
    assert "fission" not in REACTIONS
    writer_vocabulary = set(REACTIONS) | {"fission"}

    unresolvable = sorted(writer_vocabulary - set(rust))
    assert not unresolvable, (
        "the transmutation writer can emit reaction names that yani-transmute's "
        f"REACTION_MT_MAP does not resolve: {unresolvable}. These fail silently: "
        "the name maps to no MT, so the rate is never computed and the reaction "
        "vanishes from the network (#379). Add them to REACTION_MT_MAP in "
        "crates/yani-transmute/src/lib.rs."
    )


def test_the_two_sides_agree_on_which_mt_each_name_means():
    """A shared name must not mean different MTs on each side.

    Weaker than the subset check but catches the other half of the failure
    mode: a name present on both sides, pointing at different reactions.
    """
    from endf.chain import REACTIONS

    rust = _rust_reaction_mt_map()

    disagreements = []
    for name, info in REACTIONS.items():
        if name in rust and rust[name] not in info.mts:
            disagreements.append((name, rust[name], sorted(info.mts)))

    assert not disagreements, (
        "name means a different MT on each side (name, rust_mt, endf_mts): "
        f"{disagreements}"
    )
