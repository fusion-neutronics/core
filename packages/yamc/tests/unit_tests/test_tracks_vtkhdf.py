"""Particle tracks: particle type per track and ``Tracks.to_vtkhdf``."""

import pytest

import yamc

h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def tracks():
    """Tracks of a small coupled neutron-photon run (14 MeV into iron)."""
    iron = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    iron.read_nuclear_data({"Fe56": "tests/Fe56.arrow"}, photon_data={"Fe": "tests/Fe.arrow"})
    sphere = yamc.Sphere(radius=20.0, boundary="vacuum")
    model = yamc.Model(
        geometry=yamc.Geometry([yamc.Cell(region=sphere.below, material=iron)]),
        source=yamc.NeutronSource(position=(0, 0, 0), energy=14.0e6),
        transport_secondary_photons=True,
        verbose=[],
    )
    return model.simulate_transport(total_particles=20, seed=1, capture_tracks=20).tracks


def test_each_track_is_one_particle_with_its_type(tracks):
    types = {t.particle_type for t in tracks.tracks}
    assert types == {"neutron", "photon"}
    for t in tracks.tracks:
        assert {e.particle_type for e in t.events} == {t.particle_type}
        assert {e.particle_id for e in t.events} == {t.particle_id}
        assert t.events[0].event_type == ("source_birth" if t.generation == 0 else "secondary_birth")
        assert (t.parent_id is None) == (t.generation == 0)
    # Every source neutron is particle 0 of its history; photons have a parent.
    sources = [t for t in tracks.tracks if t.generation == 0]
    assert len(sources) == 20
    assert all(t.particle_id == 0 and t.particle_type == "neutron" for t in sources)
    by_key = {(t.history, t.particle_id): t for t in tracks.tracks}
    for t in tracks.tracks:
        if t.parent_id is not None:
            assert by_key[(t.history, t.parent_id)].generation == t.generation - 1


def test_dataframe_records_carry_particle_type(tracks):
    records = tracks.to_dataframe_records()
    assert len(records) == tracks.total_events()
    assert {r["particle_type"] for r in records} == {"neutron", "photon"}
    assert tracks.tracks[0].events[0].to_dict()["particle_type"] == "neutron"


def test_to_vtkhdf_round_trip(tracks, tmp_path):
    path = tmp_path / "tracks.vtkhdf"
    tracks.to_vtkhdf(path)
    with h5py.File(path, "r") as f:
        root = f["VTKHDF"]
        assert root.attrs["Type"] == b"PolyData"
        n_points, n_lines = tracks.total_events(), len(tracks)
        assert root["NumberOfPoints"][0] == n_points
        assert root["Points"].shape == (n_points, 3)
        lines = root["Lines"]
        assert lines["NumberOfCells"][0] == n_lines
        offsets = lines["Offsets"][()]
        assert offsets[0] == 0 and offsets[-1] == n_points
        assert list(np.diff(offsets)) == [len(t) for t in tracks.tracks]
        for empty in ("Vertices", "Polygons", "Strips"):
            assert root[empty]["NumberOfCells"][0] == 0

        names = [n.decode() if isinstance(n, bytes) else n for n in root.attrs["particle_type_names"]]
        codes = root["CellData"]["particle_type"][()]
        assert [names[c] for c in codes] == [t.particle_type for t in tracks.tracks]
        assert list(root["CellData"]["history"][()]) == [t.history for t in tracks.tracks]

        first = tracks.tracks[0].events
        np.testing.assert_allclose(root["Points"][: len(first)], [e.position for e in first])
        np.testing.assert_allclose(root["PointData"]["energy"][: len(first)], [e.energy_out for e in first])
        event_names = [n.decode() if isinstance(n, bytes) else n for n in root.attrs["event_type_names"]]
        assert event_names[root["PointData"]["event_type"][0]] == "source_birth"
