"""Per-nuclide tally bins (``nuclides=``) on both backends.

A specific nuclide's bin scores that nuclide's own macroscopic cross section,
``N_n * sigma_n``, so the bins of every nuclide in a material add up to the
material total. On the GPU each (score, nuclide bin) is its own kernel entry
reading its own cross-section table; the collision estimator weights each
nuclide's cross section by ``1 / Sigma_t`` at the collision, on both backends.
"""

import os

import pytest

yamc = pytest.importorskip("yamc")

TESTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "crates", "yamc", "tests"
)
NUCLIDES = ("Li6", "Li7", "Fe56")
SCORES = ["total", "(n,gamma)", "heating"]

needs_gpu = pytest.mark.skipif(
    not yamc.parallel.gpu_available(),
    reason="no GPU with f64 compute available, or yamc was built without the `gpu` feature",
)


def _model(estimator):
    """A sphere of a Li6/Li7/Fe56 mix, with one tally binned by every nuclide
    and the total, and one with no ``nuclides=`` for reference."""
    material = yamc.Material(
        composition={"Li6": 0.3, "Li7": 0.3, "Fe56": 0.4}, density=3.0, temperature=294
    )
    material.read_nuclear_data(
        {name: os.path.join(TESTS_DIR, f"{name}.arrow") for name in NUCLIDES}
    )
    sphere = yamc.Sphere(radius=20.0, boundary="vacuum")
    cell = yamc.Cell(name="mix", region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    energy_bins = [1e-5, 1e5, 1e6, 15e6]
    binned = yamc.Tally(
        cells=cell,
        scores=SCORES,
        nuclides=list(NUCLIDES) + ["total"],
        energy_bins=energy_bins,
        estimator=estimator,
        name="binned",
    )
    plain = yamc.Tally(
        cells=cell, scores=SCORES, energy_bins=energy_bins, estimator=estimator, name="plain"
    )
    source = yamc.NeutronSource(
        position=(0.0, 0.0, 0.0), energy=yamc.sources.Discrete([14.06e6], [1.0])
    )
    model = yamc.Model(geometry=geometry, tallies=[binned, plain], source=source, verbose=[])
    return model, binned, plain


def _bins(result, n_scores, n_nuclides, n_energy):
    """Reshape a tally's flat mean and std into ``[score][nuclide][energy]``."""

    def shape(flat):
        flat = list(flat)
        return [
            [
                flat[(s * n_nuclides + n) * n_energy : (s * n_nuclides + n + 1) * n_energy]
                for n in range(n_nuclides)
            ]
            for s in range(n_scores)
        ]

    return shape(result.mean), shape(result.standard_deviation)


def _check_nuclides_add_up(results, binned, plain):
    """Every nuclide's bins sum to the total bin, which equals the tally with
    no ``nuclides=``, and each nuclide's bins carry their own share."""
    n_nuc = len(NUCLIDES) + 1
    mean, _ = _bins(results[binned], len(SCORES), n_nuc, 3)
    plain_mean, _ = _bins(results[plain], len(SCORES), 1, 3)
    for s, score in enumerate(SCORES):
        for e in range(3):
            parts = [mean[s][n][e] for n in range(len(NUCLIDES))]
            total = mean[s][len(NUCLIDES)][e]
            assert total == pytest.approx(plain_mean[s][0][e], rel=1e-12, abs=1e-30), score
            assert sum(parts) == pytest.approx(total, rel=1e-6, abs=1e-20), (score, e, parts)
        # The 14 MeV bin: each nuclide carries a share of the total, not all
        # of it in one bin and nothing in the others.
        shares = [mean[s][n][2] for n in range(len(NUCLIDES))]
        assert all(0.0 < x < mean[s][len(NUCLIDES)][2] for x in shares), (score, shares)


def _assert_agree(a, b, label):
    """Two runs' ``[score][nuclide][energy]`` (mean, std) agree within 5 sigma."""
    (am, as_), (bm, bs) = a, b
    for s in range(len(am)):
        for n in range(len(am[s])):
            for e in range(len(am[s][n])):
                sigma = (as_[s][n][e] ** 2 + bs[s][n][e] ** 2) ** 0.5
                diff = abs(am[s][n][e] - bm[s][n][e])
                assert diff <= 5.0 * sigma + 1e-9 * abs(bm[s][n][e]), (
                    label,
                    SCORES[s],
                    n,
                    e,
                    am[s][n][e],
                    bm[s][n][e],
                    sigma,
                )


@pytest.mark.parametrize("estimator", ["track-length", "collision"])
def test_cpu_nuclide_bins_add_up_to_the_material(estimator):
    model, binned, plain = _model(estimator)
    results = model.simulate_transport(total_particles=20_000, seed=3)
    _check_nuclides_add_up(results, binned, plain)


def test_cpu_estimators_agree_per_nuclide():
    """The collision estimator used to put the whole material's rate into the
    first nuclide's bin and nothing into the others."""
    n_nuc = len(NUCLIDES) + 1
    out = {}
    for estimator in ("track-length", "collision"):
        model, binned, _ = _model(estimator)
        results = model.simulate_transport(total_particles=40_000, seed=5)
        out[estimator] = _bins(results[binned], len(SCORES), n_nuc, 3)
    _assert_agree(out["track-length"], out["collision"], "track-length vs collision")


@needs_gpu
@pytest.mark.parametrize("estimator", ["track-length", "collision"])
def test_gpu_nuclide_bins_match_the_cpu(estimator):
    model, binned, plain = _model(estimator)
    gpu = model.simulate_transport(total_particles=40_000, seed=3, compute="gpu")
    _check_nuclides_add_up(gpu, binned, plain)
    cpu = model.simulate_transport(total_particles=40_000, seed=7)
    n_nuc = len(NUCLIDES) + 1
    _assert_agree(
        _bins(gpu[binned], len(SCORES), n_nuc, 3),
        _bins(cpu[binned], len(SCORES), n_nuc, 3),
        f"gpu vs cpu ({estimator})",
    )
