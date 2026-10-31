"""``compute='gpu'`` and per-nuclide tally bins (``nuclides=``).

The GPU kernel scores the cell material's macroscopic cross section, and its
writeback put that into the first nuclide bin. A ``nuclides=["Li6", "Li7",
"Fe56"]`` tally therefore came back with the whole material's rate under Li6
and zero under Li7 and Fe56, with no diagnostic. It is refused now, and the
material total (no ``nuclides=``, or ``nuclides=["total"]``) still runs.
"""

import os

import pytest

yamc = pytest.importorskip("yamc")

TESTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "crates", "yamc", "tests"
)
NUCLIDES = ("Li6", "Li7", "Fe56")

needs_gpu = pytest.mark.skipif(
    not yamc.parallel.gpu_available(),
    reason="no GPU with f64 compute available, or yamc was built without the `gpu` feature",
)


def _model(tallied_nuclides, load_data=False):
    """A sphere of a Li6/Li7/Fe56 mix with one tally per entry of
    ``tallied_nuclides`` (``None`` for a tally with no ``nuclides=``)."""
    material = yamc.Material(
        composition={"Li6": 0.3, "Li7": 0.3, "Fe56": 0.4}, density=3.0, temperature=294
    )
    if load_data:
        material.read_nuclear_data(
            {name: os.path.join(TESTS_DIR, f"{name}.arrow") for name in NUCLIDES}
        )
    sphere = yamc.Sphere(radius=20.0, boundary="vacuum")
    cell = yamc.Cell(name="mix", region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    tallies = [
        yamc.Tally(
            cells=cell,
            scores=["total", 102],
            name=f"t{i}",
            **({} if nuclides is None else {"nuclides": nuclides}),
        )
        for i, nuclides in enumerate(tallied_nuclides)
    ]
    source = yamc.NeutronSource(
        position=(0.0, 0.0, 0.0), energy=yamc.sources.Discrete([14.06e6], [1.0])
    )
    return yamc.Model(geometry=geometry, tallies=tallies, source=source, verbose=[]), tallies


@pytest.mark.parametrize(
    "nuclides",
    [list(NUCLIDES), ["Fe56"], ["Li6", "total"]],
    ids=["three-nuclides", "one-nuclide", "nuclide-and-total"],
)
def test_the_gpu_refuses_per_nuclide_bins(nuclides):
    # Checked before any nuclear data is loaded or any GPU is initialized, so
    # this runs on any machine.
    model, _ = _model([nuclides])
    with pytest.raises(ValueError, match=r"tally 0: per-nuclide tally bins \(nuclides=\)"):
        model.simulate_transport(total_particles=100, compute="gpu")


@needs_gpu
def test_the_material_total_bin_still_runs_on_the_gpu():
    # The kernel's score is the material total, so `nuclides=["total"]` is the
    # same tally as no `nuclides=` at all: identical on the GPU, and within
    # statistics of the CPU.
    model, (plain, total_bin) = _model([None, ["total"]], load_data=True)
    gpu = model.simulate_transport(total_particles=20_000, seed=3, compute="gpu")
    assert list(gpu[total_bin].mean) == list(gpu[plain].mean)
    cpu = model.simulate_transport(total_particles=20_000, seed=3)
    for g, c, gs, cs in zip(
        gpu[total_bin].mean,
        cpu[total_bin].mean,
        gpu[total_bin].standard_deviation,
        cpu[total_bin].standard_deviation,
    ):
        assert abs(g - c) < 5.0 * (gs**2 + cs**2) ** 0.5, (g, c, gs, cs)
