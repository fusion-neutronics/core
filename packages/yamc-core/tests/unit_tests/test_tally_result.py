"""Tests for TallyResult returned from model.simulate_transport()."""

import math
import pytest
import yamc


def _keywords_available():
    try:
        m = yamc.Material(
            composition={"Li6": 1.0},
            density=1.0,
            temperature=294,
        )
        m.read_nuclear_data("endf-b8.1")
        return True
    except Exception:
        return False


requires_keywords = pytest.mark.skipif(
    not _keywords_available(),
    reason="keyword download requires download feature",
)


def _make_model(tally, total_particles=2000):
    """Build a minimal model with one Li sphere and the given tally."""
    sphere = yamc.Sphere(
        x0=0.0, y0=0.0, z0=0.0, radius=10.0, boundary="vacuum",
    )
    material = yamc.Material(
        composition={"Li6": 0.5, "Li7": 0.5},
        density=0.5,
        temperature=294,
    )
    cell = yamc.Cell(name="sphere", region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1]),
        position=(0, 0, 0),
    )
    model = yamc.Model(
        geometry=geometry,
        tallies=[tally],
        source=source,
    )
    return model, {"total_particles": total_particles, "seed": 42}


@requires_keywords
def test_single_score_no_filters():
    """1 score, cell tally -> shape (1,), dims ("score",)."""
    yamc.set_cross_section_data_entry("fendl-3.2d")
    tally = yamc.Tally(scores=["flux"])
    model, run_kwargs = _make_model(tally)
    results = model.simulate_transport(**run_kwargs)

    result = results[tally]
    assert tuple(result.shape) == (1,)
    assert tuple(result.dim_labels) == ("score",)
    assert len(result.mean) == 1
    assert len(result.standard_deviation) == 1
    assert len(result.relative_error) == 1


@requires_keywords
def test_with_energy_bins():
    """1 score + 3 energy bins -> shape (1, 3), dims ("score", "energy")."""
    yamc.set_cross_section_data_entry("fendl-3.2d")
    tally = yamc.Tally(
        scores=["flux"],
        energy_bins=[0.0, 1e6, 10e6, 20e6],
    )
    model, run_kwargs = _make_model(tally)
    results = model.simulate_transport(**run_kwargs)

    result = results[tally]
    assert tuple(result.shape) == (1, 3)
    assert tuple(result.dim_labels) == ("score", "energy")
    assert len(result.mean) == 3


@requires_keywords
def test_with_mesh():
    """1 score + mesh 3x4x5 -> shape (1, 5, 4, 3), dims with mesh_z/y/x."""
    yamc.set_cross_section_data_entry("fendl-3.2d")
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-8.0, -8.0, -8.0],
        upper_right=[8.0, 8.0, 8.0],
        shape=[3, 4, 5],
    )
    tally = yamc.Tally(scores=["flux"], mesh=mesh)
    model, run_kwargs = _make_model(tally)
    results = model.simulate_transport(**run_kwargs)

    result = results[tally]
    assert tuple(result.shape) == (1, 5, 4, 3)
    assert tuple(result.dim_labels) == ("score", "mesh_z", "mesh_y", "mesh_x")
    assert len(result.mean) == 1 * 5 * 4 * 3


@requires_keywords
def test_with_mesh_and_energy():
    """1 score + 2 energy bins + mesh 3x4x5."""
    yamc.set_cross_section_data_entry("fendl-3.2d")
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-8.0, -8.0, -8.0],
        upper_right=[8.0, 8.0, 8.0],
        shape=[3, 4, 5],
    )
    tally = yamc.Tally(
        scores=["flux"],
        mesh=mesh,
        energy_bins=[0.0, 1e6, 20e6],
    )
    model, run_kwargs = _make_model(tally)
    results = model.simulate_transport(**run_kwargs)

    result = results[tally]
    assert tuple(result.shape) == (1, 2, 5, 4, 3)
    assert tuple(result.dim_labels) == ("score", "energy", "mesh_z", "mesh_y", "mesh_x")
    assert len(result.mean) == 1 * 2 * 5 * 4 * 3


@requires_keywords
def test_multiple_scores():
    """2 scores + mesh -> shape[0] == 2."""
    yamc.set_cross_section_data_entry("fendl-3.2d")
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-8.0, -8.0, -8.0],
        upper_right=[8.0, 8.0, 8.0],
        shape=[2, 2, 2],
    )
    tally = yamc.Tally(scores=["flux", "heating"], mesh=mesh)
    model, run_kwargs = _make_model(tally)
    results = model.simulate_transport(**run_kwargs)

    result = results[tally]
    assert result.shape[0] == 2
    assert result.dim_labels[0] == "score"
    assert len(result.mean) == 2 * 2 * 2 * 2


@requires_keywords
def test_with_nuclides():
    """1 score + nuclides ["Li6", "total"] -> shape (1, 2), dims with nuclide."""
    yamc.set_cross_section_data_entry("fendl-3.2d")
    tally = yamc.Tally(
        scores=["H3-production"],
        nuclides=["Li6", "total"],
    )
    model, run_kwargs = _make_model(tally)
    results = model.simulate_transport(**run_kwargs)

    result = results[tally]
    assert tuple(result.shape) == (1, 2)
    assert tuple(result.dim_labels) == ("score", "nuclide")
    assert len(result.mean) == 2


@requires_keywords
def test_shape_product_matches_data_length():
    """product(shape) == len(mean) for a multi-dimensional tally."""
    yamc.set_cross_section_data_entry("fendl-3.2d")
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-8.0, -8.0, -8.0],
        upper_right=[8.0, 8.0, 8.0],
        shape=[2, 3, 4],
    )
    tally = yamc.Tally(
        scores=["flux"],
        mesh=mesh,
        energy_bins=[0.0, 1e6, 20e6],
    )
    model, run_kwargs = _make_model(tally)
    results = model.simulate_transport(**run_kwargs)

    result = results[tally]
    expected = math.prod(result.shape)
    assert expected == len(result.mean)
    assert expected == len(result.standard_deviation)
    assert expected == len(result.relative_error)


@requires_keywords
def test_dim_labels_are_strings():
    """All dim_labels elements must be str."""
    yamc.set_cross_section_data_entry("fendl-3.2d")
    tally = yamc.Tally(scores=["flux"])
    model, run_kwargs = _make_model(tally)
    results = model.simulate_transport(**run_kwargs)

    result = results[tally]
    for label in result.dim_labels:
        assert isinstance(label, str)


@requires_keywords
def test_shape_matches_dim_labels():
    """len(shape) == len(dim_labels)."""
    yamc.set_cross_section_data_entry("fendl-3.2d")
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-8.0, -8.0, -8.0],
        upper_right=[8.0, 8.0, 8.0],
        shape=[2, 2, 2],
    )
    tally = yamc.Tally(
        scores=["flux", "heating"],
        mesh=mesh,
        energy_bins=[0.0, 1e6, 20e6],
    )
    model, run_kwargs = _make_model(tally)
    results = model.simulate_transport(**run_kwargs)

    result = results[tally]
    assert len(result.shape) == len(result.dim_labels)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
