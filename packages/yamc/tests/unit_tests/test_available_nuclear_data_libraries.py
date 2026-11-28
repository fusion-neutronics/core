"""available_nuclear_data_libraries(): what each hosted library provides.

Read from each release's manifest, so it needs the data origin (or a cached
release). Every list is complete: an empty one says the library has none.
"""

import pytest
import yamc

KEYS = {
    "release",
    "offline",
    "neutron",
    "neutron_covariances",
    "thermal_scattering",
    "photon",
    "photonuclear",
    "decay",
    "reactions",
    "fission_yields",
    "isomeric_branching",
    "sources",
    "error",
}
COVARIANCES = {
    "cross_section",
    "resonance_parameters",
    "angular_distribution",
    "nubar",
    "fission_spectrum",
}


@pytest.fixture(scope="module")
def libraries():
    libraries = yamc.available_nuclear_data_libraries()
    if all(entry["error"] for entry in libraries.values()):
        pytest.skip(
            "no library could be resolved: the data origin is unreachable and nothing is cached"
        )
    return libraries


def test_every_hosted_library_is_listed_with_every_key(libraries):
    assert {
        "tendl-2025",
        "endf-b8.1",
        "jeff-4.0",
        "jendl-5.0",
        "fendl-3.2d",
        "irdff-ii",
        "ukdd-2012",
        "ukdd-2020",
        "ukfy-4.2",
    } <= set(libraries)
    for keyword, entry in libraries.items():
        assert set(entry) == KEYS, keyword
        if entry["error"] is None:
            assert set(entry["neutron_covariances"]) == COVARIANCES, keyword
            assert all(
                set(n) <= set(entry["neutron"])
                for n in entry["neutron_covariances"].values()
            )
        else:
            # Unresolved is None throughout, never an empty answer.
            assert all(entry[k] is None for k in KEYS - {"error"}), keyword


def test_photo_atomic_and_photonuclear_are_told_apart(libraries):
    endf, tendl = libraries["endf-b8.1"], libraries["tendl-2025"]
    if endf["error"] is None:
        assert "Fe" in endf["photon"]
        assert endf["decay"] and endf["fission_yields"]
        assert "Fe56" in endf["neutron_covariances"]["cross_section"]
    if tendl["error"] is None:
        assert tendl["photon"] == [], "TENDL publishes no photo-atomic data"
        assert "Fe56" in tendl["photonuclear"]
        assert not tendl["decay"]


def test_a_branching_only_library_has_nothing_else(libraries):
    irdff = libraries["irdff-ii"]
    if irdff["error"] is not None:
        pytest.skip(irdff["error"])
    assert irdff["isomeric_branching"]
    assert (
        irdff["neutron"] == [] and irdff["photon"] == [] and irdff["photonuclear"] == []
    )
    assert not (irdff["decay"] or irdff["reactions"] or irdff["fission_yields"])


@pytest.mark.parametrize(
    "keyword, part",
    [("ukdd-2012", "decay"), ("ukdd-2020", "decay"), ("ukfy-4.2", "fission_yields")],
)
def test_a_ukaea_library_has_its_one_subsection(libraries, keyword, part):
    entry = libraries[keyword]
    if entry["error"] is not None:
        pytest.skip(entry["error"])
    assert entry[part]
    assert entry["neutron"] == [] and entry["photon"] == [] and entry["photonuclear"] == []
    others = {"decay", "reactions", "fission_yields"} - {part}
    assert not any(entry[k] for k in others) and not entry["isomeric_branching"]
