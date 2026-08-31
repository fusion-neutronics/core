"""Unit tests for synthesis algorithms."""

import numpy as np
import pytest

from nuclear_data_to_arrow.synthesis import (
    is_scattering_mt,
    is_fission_mt,
    synthesize_hierarchical_mts,
    build_fast_xs,
    N_LOG_BINS,
    SYNTHETIC_MTS,
    ALL_SCATTERING_MTS,
    FISSION_MTS,
    ABSORPTION_MTS,
)


class TestMTClassification:
    """Test MT classification functions."""

    def test_elastic_is_scattering(self):
        assert is_scattering_mt(2)

    def test_inelastic_range_is_scattering(self):
        for mt in range(50, 92):
            assert is_scattering_mt(mt), f"MT {mt} should be scattering"

    def test_non_inelastic_scattering(self):
        for mt in [5, 11, 16, 17, 22, 23, 24, 25, 28, 29, 30]:
            assert is_scattering_mt(mt), f"MT {mt} should be scattering"

    def test_fission_not_scattering(self):
        for mt in [18, 19, 20, 21, 38]:
            assert not is_scattering_mt(mt), f"MT {mt} should not be scattering"

    def test_capture_not_scattering(self):
        assert not is_scattering_mt(102)

    def test_fission_mts(self):
        for mt in [18, 19, 20, 21, 38]:
            assert is_fission_mt(mt), f"MT {mt} should be fission"

    def test_elastic_not_fission(self):
        assert not is_fission_mt(2)

    def test_absorption_mts_not_fission(self):
        for mt in [102, 103, 104]:
            assert not is_fission_mt(mt)


class TestSynthesis:
    """Test MT synthesis with real nuclear data."""

    def test_synthesis_returns_expected_mts(self, neutron_data):
        data = neutron_data
        temp = list(data.temperatures)[0]
        result = synthesize_hierarchical_mts(data, temp)

        assert 1 in result, "MT 1 (total) must always be present"
        assert 3 in result, "MT 3 (non-elastic) must be present"
        assert 4 in result, "MT 4 (inelastic) must be present"
        assert 27 in result, "MT 27 (absorption) must be present"
        assert 101 in result, "MT 101 (disappearance) must always be present"

    def test_mt1_equals_mt2_plus_mt3(self, neutron_data):
        data = neutron_data
        temp = list(data.temperatures)[0]
        result = synthesize_hierarchical_mts(data, temp)
        energy_grid = np.asarray(data.energy[temp])

        # Get elastic XS
        if 2 in data.reactions:
            from nuclear_data_to_arrow.synthesis import _interp_xs_to_grid
            elastic = _interp_xs_to_grid(data.reactions[2], temp, energy_grid)
        else:
            elastic = np.zeros(len(energy_grid))

        np.testing.assert_allclose(
            result[1], elastic + result[3],
            rtol=1e-14,
            err_msg="MT 1 != MT 2 + MT 3"
        )

    def test_mt1_nonnegative(self, neutron_data):
        data = neutron_data
        temp = list(data.temperatures)[0]
        result = synthesize_hierarchical_mts(data, temp)
        assert np.all(result[1] >= 0), "Total XS (MT 1) must be non-negative"

    def test_mt101_nonnegative(self, neutron_data):
        data = neutron_data
        temp = list(data.temperatures)[0]
        result = synthesize_hierarchical_mts(data, temp)
        assert np.all(result[101] >= 0), "MT 101 must be non-negative"


class TestFastXS:
    """Test FastXSGrid construction."""

    def test_fast_xs_structure(self, neutron_data):
        data = neutron_data
        temp = list(data.temperatures)[0]
        synth = synthesize_hierarchical_mts(data, temp)
        fxs = build_fast_xs(data, temp, synth)

        assert "log_e_min" in fxs
        assert "inv_log_delta" in fxs
        assert "log_grid_index" in fxs
        assert "xs" in fxs
        assert "energy" in fxs
        assert "scatter_mt_numbers" in fxs
        assert "scatter_mt_xs" in fxs
        assert "fission_mt_numbers" in fxs
        assert "fission_mt_xs" in fxs
        assert "has_partial_fission" in fxs
        assert "xs_ngamma" in fxs
        assert "photon_prod" in fxs

    def test_log_grid_index_shape(self, neutron_data):
        data = neutron_data
        temp = list(data.temperatures)[0]
        synth = synthesize_hierarchical_mts(data, temp)
        fxs = build_fast_xs(data, temp, synth)
        assert fxs["log_grid_index"].shape == (N_LOG_BINS + 1,)

    def test_log_grid_index_monotonic(self, neutron_data):
        data = neutron_data
        temp = list(data.temperatures)[0]
        synth = synthesize_hierarchical_mts(data, temp)
        fxs = build_fast_xs(data, temp, synth)
        lgi = fxs["log_grid_index"]
        assert np.all(lgi[1:] >= lgi[:-1]), "log_grid_index must be non-decreasing"

    def test_xs_shape(self, neutron_data):
        data = neutron_data
        temp = list(data.temperatures)[0]
        n_energy = len(data.energy[temp])
        synth = synthesize_hierarchical_mts(data, temp)
        fxs = build_fast_xs(data, temp, synth)
        assert fxs["xs"].shape == (n_energy, 4)

    def test_total_equals_components(self, neutron_data):
        data = neutron_data
        temp = list(data.temperatures)[0]
        synth = synthesize_hierarchical_mts(data, temp)
        fxs = build_fast_xs(data, temp, synth)

        total = fxs["xs"][:, 0]
        absorption = fxs["xs"][:, 1]
        scattering = fxs["xs"][:, 2]
        fission = fxs["xs"][:, 3]

        np.testing.assert_allclose(
            total, absorption + scattering + fission,
            rtol=1e-10, atol=1e-30,
            err_msg="total != absorption + scattering + fission"
        )


# ---------------------------------------------------------------------------
# MT 101 membership (issue #23)
# ---------------------------------------------------------------------------

class _FakeXS:
    """Minimal stand-in for a Tabulated1D on the full energy grid."""

    def __init__(self, y):
        self.y = np.asarray(y, dtype=float)
        self._threshold_idx = 0


class _FakeReaction:
    def __init__(self, y, temperature):
        self.xs = {temperature: _FakeXS(y)}


class _FakeNeutron:
    """A nuclide carrying only the MTs a test needs, on a 4-point grid."""

    def __init__(self, mts, temperature="294K", n=4):
        self.energy = {temperature: np.linspace(1.0, 4.0, n)}
        self.reactions = {mt: _FakeReaction([v] * n, temperature)
                          for mt, v in mts.items()}


# The neutron-emitting channels that were wrongly summed into MT 101. Every one
# re-emits a neutron, so none is a disappearance reaction.
_NEUTRON_EMITTING = [183, 184, 185, 186, 187, 188, 189, 190,
                     194, 195, 196, 198, 199]


@pytest.mark.parametrize("mt", _NEUTRON_EMITTING)
def test_neutron_emitting_channels_are_not_absorption(mt):
    """MT 101 is neutron disappearance, so a channel that re-emits a neutron
    must not contribute to it."""
    assert mt not in ABSORPTION_MTS


@pytest.mark.parametrize("mt", _NEUTRON_EMITTING)
def test_neutron_emitting_channels_are_scattering(mt):
    """They do belong in the scattering classification, which is what makes
    double counting possible if they are also in MT 101."""
    assert is_scattering_mt(mt)


def test_mt110_is_not_absorption():
    """MT 110 is not a defined ENDF reaction and was in the old private list."""
    assert 110 not in ABSORPTION_MTS


def test_absorption_and_scattering_do_not_overlap():
    """The invariant that prevents double counting: MT 3 is built as
    scattering + fission + MT 101, so no MT may be in two of those groups."""
    assert not (ABSORPTION_MTS & ALL_SCATTERING_MTS)
    assert not (ABSORPTION_MTS & FISSION_MTS)
    assert not (ALL_SCATTERING_MTS & FISSION_MTS)


def test_neutron_emitting_channel_is_counted_once_in_mt3():
    """A nuclide with a neutron-emitting multi-particle channel, which is the
    La139 case. Elastic 2.0, (n,npd) 0.5, capture 1.0. MT 3 is everything but
    elastic, so it must be 1.5, not 2.0 with (n,npd) counted twice.
    """
    T = "294K"
    data = _FakeNeutron({2: 2.0, 183: 0.5, 102: 1.0}, temperature=T)
    result = synthesize_hierarchical_mts(data, T)

    np.testing.assert_allclose(result[101], 1.0)   # capture only
    np.testing.assert_allclose(result[27], 1.0)    # no fission
    np.testing.assert_allclose(result[3], 1.5)     # (n,npd) + capture, once
    np.testing.assert_allclose(result[1], 3.5)     # elastic + MT 3


def test_total_equals_sum_of_partials():
    """With no channel double counted, MT 1 is the sum of every partial."""
    T = "294K"
    partials = {2: 2.0, 4: 0.0, 16: 0.25, 102: 1.0, 103: 0.125,
                183: 0.5, 190: 0.0625, 199: 0.03125}
    data = _FakeNeutron(partials, temperature=T)
    result = synthesize_hierarchical_mts(data, T)
    expected = sum(v for mt, v in partials.items() if mt not in SYNTHETIC_MTS)
    np.testing.assert_allclose(result[1], expected)
