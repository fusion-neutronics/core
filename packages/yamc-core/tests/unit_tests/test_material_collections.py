"""Tests for the curated material collections (yamc.materials)."""

import pytest

import yamc

pnnl = yamc.materials.pnnl


def test_collections_lists_pnnl():
    assert yamc.materials.collections() == ["pnnl"]


def test_dotted_import_works():
    import yamc.materials

    assert yamc.materials.pnnl is pnnl


def test_length_is_the_bundled_count():
    # 411 entries in PNNL-15870 Rev. 2, less the self-contradictory Fe2B one.
    assert len(pnnl) == 410
    assert len(pnnl.names()) == 410


def test_lookup_returns_a_material():
    steel = pnnl["Steel, Stainless 304"]
    assert isinstance(steel, yamc.Material)
    assert steel.name == "Steel, Stainless 304"
    assert steel.density == pytest.approx(8.03)
    assert steel.density_units == "g/cm3"


def test_lookup_returns_a_fresh_object_each_time():
    """Mutating one lookup must not affect the collection or another."""
    a = pnnl["Steel, Stainless 304"]
    a.name = "firstwall_material"
    b = pnnl["Steel, Stainless 304"]
    assert b.name == "Steel, Stainless 304"
    assert a.name == "firstwall_material"


def test_material_can_be_renamed_for_mesh_geometry():
    steel = pnnl["Steel, Stainless 304"]
    steel.name = "firstwall_material"
    assert steel.name == "firstwall_material"


def test_natural_elements_expand_to_yamc_abundances():
    steel = pnnl["Steel, Stainless 304"]
    names = [n for n, _ in steel.nuclides]
    assert "Fe56" in names and "Fe54" in names
    assert "Cr52" in names


def test_enriched_entries_keep_their_own_isotopics():
    comp = pnnl.entry("Water, Heavy")["composition"]
    assert "H2" in comp and "H" not in comp
    heavy = pnnl["Water, Heavy"]
    names = [n for n, _ in heavy.nuclides]
    assert "H2" in names
    assert "H1" not in names


def test_search_is_case_insensitive_substring():
    hits = pnnl.search("stainless")
    assert "Steel, Stainless 304" in hits
    assert hits == pnnl.search("STAINLESS")
    assert pnnl.search("no such material anywhere") == []


def test_search_results_can_be_looked_up_directly():
    for name in pnnl.search("concrete"):
        assert isinstance(pnnl[name], yamc.Material)


def test_contains_and_iteration():
    assert "Steel, Stainless 304" in pnnl
    assert "Steel, Stainless 999" not in pnnl
    names = list(pnnl)
    assert len(names) == 410
    assert names == pnnl.names()


def test_unknown_name_raises_keyerror_with_suggestions():
    with pytest.raises(KeyError) as exc:
        pnnl["Steel, Stainless 999"]
    message = str(exc.value)
    assert "did you mean" in message
    assert "Stainless" in message


def test_entry_carries_provenance():
    e = pnnl.entry("Steel, Stainless 304")
    assert e["number"] == 331
    assert e["citation"] == "PNNL-15870, Rev. 2"
    assert e["url"].endswith("PNNL-15870Rev2.pdf")
    assert e["composition"]["Fe"] == pytest.approx(0.667971)


def test_entry_composition_is_atom_fractions_summing_to_one():
    for name in pnnl.names():
        comp = pnnl.entry(name)["composition"]
        assert comp, f"{name}: empty composition"
        assert sum(comp.values()) == pytest.approx(1.0, abs=5e-3), name


def test_material_applies_overrides():
    m = pnnl.material(
        "Concrete, Ordinary (NIST)",
        name="bioshield",
        density=1.8,
        temperature=600,
        volume=100.0,
        transmutable=True,
        id=7,
    )
    assert m.name == "bioshield"
    assert m.density == pytest.approx(1.8)
    assert m.temperature == "600"
    assert m.volume == pytest.approx(100.0)
    assert m.transmutable is True
    assert m.id == 7


def test_material_without_overrides_matches_subscript():
    a = pnnl.material("Steel, Stainless 304")
    b = pnnl["Steel, Stainless 304"]
    assert a.name == b.name
    assert a.density == pytest.approx(b.density)
    assert a.nuclides == b.nuclides


def test_material_rejects_a_bad_density():
    for bad in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError):
            pnnl.material("Steel, Stainless 304", density=bad)


def test_material_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        pnnl.material("Not A Material")


def test_the_contradictory_iron_boride_entry_is_omitted():
    assert "Iron Boride (Fe2B)" not in pnnl
    # The unaffected FeB entry is still there.
    assert pnnl.entry("Iron Boride (FeB)")["number"] == 174


def test_every_entry_builds_a_material():
    """The whole collection must be constructible, not just the spot checks."""
    for name in pnnl.names():
        m = pnnl[name]
        assert m.name == name
        assert m.nuclides, f"{name}: no nuclides"
        assert m.density > 0.0


def test_collection_repr_names_the_source():
    text = repr(pnnl)
    assert "410" in text
    assert "PNNL-15870" in text


def test_citation_and_url_are_exposed():
    assert pnnl.citation == "PNNL-15870, Rev. 2"
    assert pnnl.url.startswith("https://")
