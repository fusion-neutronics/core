"""Fission energy release emission for the delayed-photon scaling (yamc issue #369).

A fission releases photons from the fission products' decay as well as promptly,
and the evaluation's prompt photon production does not include them. OpenMC scales
fission photon production by f(E) = (prompt + delayed) / prompt; without these two
terms in the Arrow output yamc cannot, and actinide fission photon production comes
out ~38% low.

The terms are emitted as the functions the evaluation stores rather than as values
on an energy grid, because neither is necessarily a polynomial. These tests pin
that both representations survive, and -- more importantly -- that anything yamc
cannot evaluate exactly fails the conversion loudly instead of being written out
and silently mis-evaluated.
"""

import numpy as np
import pytest
from endf.function import Tabulated1D
from numpy.polynomial import Polynomial

from nuclear_data_to_arrow.neutron_writer import _fission_photon_rows


class _Release:
    def __init__(self, prompt, delayed):
        self.prompt_photons = prompt
        self.delayed_photons = delayed


class _Data:
    def __init__(self, name, release):
        self.name = name
        self.fission_energy = release


def _rows(prompt, delayed, name="U235"):
    return _fission_photon_rows(_Data(name, _Release(prompt, delayed)))


def _by_role(rows):
    return {r["role"]: r for r in rows}


def test_no_fission_energy_release_emits_nothing():
    """The common case: 478 of 557 ENDF/B-VIII.1 nuclides have no such data.

    They must produce no section at all, so yamc leaves the scaling off and takes
    its f = 1.0 branch, exactly as before the feature existed.
    """
    assert _fission_photon_rows(_Data("Fe56", None)) == []


def test_polynomial_terms_round_trip_ascending():
    """76 of the 79 have both terms as polynomials.

    Coefficient ORDER is the thing to pin: yamc evaluates with Horner over
    ascending powers, so emitting them descending would silently produce a wrong
    but plausible-looking factor.
    """
    rows = _by_role(_rows(Polynomial([6.824183e6, 1.693e-2]),
                          Polynomial([3.7543e6, -7.5e-2])))
    prompt = rows["prompt_photons"]
    assert prompt["kind"] == "polynomial"
    assert prompt["coefficients"][0] == pytest.approx(6.824183e6)
    assert prompt["coefficients"][1] == pytest.approx(1.693e-2)
    assert rows["delayed_photons"]["coefficients"][1] == pytest.approx(-7.5e-2)


def test_tabulated_prompt_is_preserved_with_its_interpolation():
    """U235, U238 and Pu239 tabulate the prompt term.

    The x/y arrays and the raw ENDF interpolation/breakpoint arrays all have to
    reach yamc, so it can refuse a scheme it cannot evaluate rather than guessing.
    """
    tab = Tabulated1D([1.0e-5, 1.0e6, 2.0e6], [7.28e6, 7.97e6, 7.99e6],
                      breakpoints=[3], interpolation=[2])
    rows = _by_role(_rows(tab, Polynomial([6.33e6, -7.5e-2])))
    prompt = rows["prompt_photons"]
    assert prompt["kind"] == "tabulated"
    assert prompt["x"] == pytest.approx([1.0e-5, 1.0e6, 2.0e6])
    assert prompt["y"] == pytest.approx([7.28e6, 7.97e6, 7.99e6])
    assert prompt["interpolation"] == [2]
    assert prompt["breakpoints"] == [3]
    # The two terms are independent: a tabulated prompt sits beside a polynomial
    # delayed, which is exactly why a single shared representation would not do.
    assert rows["delayed_photons"]["kind"] == "polynomial"


@pytest.mark.parametrize("scheme", [1, 3, 4, 5])
def test_unsupported_interpolation_scheme_fails_the_conversion(scheme):
    """Histogram/log schemes must raise, not be written out as if linear.

    Silently treating a log-log region as linear would bias fission photon
    production with nothing to show for it -- the same class of silent error that
    let the missing scaling go unnoticed in the first place.
    """
    tab = Tabulated1D([1.0, 2.0], [1.0, 2.0], breakpoints=[2], interpolation=[scheme])
    with pytest.raises(ValueError, match="interpolation scheme"):
        _rows(tab, Polynomial([1.0]))


def test_multi_region_table_fails_the_conversion():
    """No published table is multi-region, so this is a tripwire for new data."""
    tab = Tabulated1D([1.0, 2.0, 3.0], [1.0, 2.0, 3.0],
                      breakpoints=[2, 3], interpolation=[2, 2])
    with pytest.raises(ValueError, match="multi-region"):
        _rows(tab, Polynomial([1.0]))


def test_unknown_function_type_fails_the_conversion():
    """Any other Function1D (Sum, Regions1D, ...) must stop the build."""
    class _Weird:
        pass

    with pytest.raises(ValueError, match="cannot evaluate"):
        _rows(_Weird(), Polynomial([1.0]))


def test_missing_term_fails_the_conversion():
    """Both terms are needed to form the ratio; half of one is a data bug."""
    with pytest.raises(ValueError, match="both terms"):
        _rows(Polynomial([1.0]), None)


def test_emitted_values_reproduce_the_source_functions_scaling():
    """End to end: the emitted numbers must reproduce f(E) from the source objects.

    Uses the real U235 terms (tabulated prompt, polynomial delayed) and checks the
    ratio rebuilt from the EMITTED columns against a direct evaluation of the same
    functions. 1.7734 is the value OpenMC reports for U235 at 2 MeV, so the final
    assertion is still a cross-code check even though openmc is not imported.
    """
    prompt = Tabulated1D([1.0e-5, 1.0e6, 2.0e6, 3.0e6],
                         [7.281253e6, 7.969900e6, 7.990447e6, 8.461902e6],
                         breakpoints=[4], interpolation=[2])
    delayed = Polynomial([6.33e6, -7.5e-2])
    rows = _by_role(_rows(prompt, delayed))

    energy = 2.0e6
    # Rebuild f(E) from what was EMITTED, the way yamc will.
    x = np.asarray(rows["prompt_photons"]["x"])
    y = np.asarray(rows["prompt_photons"]["y"])
    emitted_prompt = np.interp(energy, x, y)
    coeffs = rows["delayed_photons"]["coefficients"]
    emitted_delayed = sum(c * energy ** i for i, c in enumerate(coeffs))
    emitted_f = (emitted_prompt + emitted_delayed) / emitted_prompt

    expected_f = (float(prompt(energy)) + float(delayed(energy))) / float(prompt(energy))
    assert emitted_f == pytest.approx(expected_f, rel=1e-12)
    # Sanity: this is a large correction, not a nudge.
    assert emitted_f == pytest.approx(1.7734, abs=1e-4)
