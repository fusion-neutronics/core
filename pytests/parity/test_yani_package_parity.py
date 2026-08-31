"""The two wheels must be the same code, not merely similar code.

``yani`` is built from the same ``yani-python`` crate that ``yamc`` links, so a
material and schedule built through either package has to step to the same
inventory, bit for bit. That is the guard that splitting the bindings out
(issue #381) changed no physics.

Skipped unless the standalone ``yani`` wheel is installed alongside ``yamc``;
``maturin develop -m packages/yani-core/pyproject.toml`` puts it there.
"""

from pathlib import Path

import pytest

import yamc

yani = pytest.importorskip("yani", reason="standalone yani wheel not installed")

_CHAIN = str(
    Path(__file__).resolve().parents[2] / "tests" / "transmutation-endf-b8.1-sfr.arrow"
)
_LIBRARY = "endf-b8.1"


def _schedule(pkg, source):
    return pkg.PulseSchedule(
        [
            pkg.Pulse(rate=1.11e14, duration=(30, "d"), source=source),
            pkg.Cooldown(duration=(1, "d")),
        ]
    )


def _inventory(pkg, chain_dir, cross_sections):
    pkg.cross_section_data = cross_sections
    pkg.transmutation_decay_data = chain_dir
    pkg.transmutation_reactions = chain_dir
    pkg.transmutation_fission_yields = chain_dir
    material = pkg.Material(
        composition={"Fe56": 1.0},
        density=7.874,
        temperature=294,
        name="iron",
        volume=1000.0,
    )
    spectrum = pkg.NeutronSource(
        energy=pkg.sources.Histogram([1e-5, 1e5, 1e6, 1.5e7], [1e12, 1e13, 1e14])
    )
    results = material.transmute(schedule=_schedule(pkg, spectrum))
    last = results.get_final_material(material.id or 0)
    return dict(last.nuclides), last.activity(), last.decay_heat()


def test_yani_and_yamc_step_to_the_same_inventory():
    yamc_inv, yamc_act, yamc_heat = _inventory(yamc, _CHAIN, _LIBRARY)
    yani_inv, yani_act, yani_heat = _inventory(yani, _CHAIN, _LIBRARY)

    # The inventory is the result, and it must match to the bit.
    assert set(yamc_inv) == set(yani_inv)
    for name, density in yamc_inv.items():
        assert yani_inv[name] == density, f"{name} differs between the two packages"
    # `activity()` and `decay_heat()` sum that inventory over a HashMap, and
    # the two packages hold separate maps whose iteration order need not agree,
    # so their last bit is summation order rather than physics.
    assert yani_act == pytest.approx(yamc_act, rel=1e-12)
    assert yani_heat == pytest.approx(yamc_heat, rel=1e-12)


def test_yani_carries_no_transport_api():
    """The point of the second wheel is what it leaves out."""
    for absent in ("Model", "Geometry", "Cell", "Tally", "Sphere", "Region"):
        assert not hasattr(yani, absent), f"yani should not expose {absent}"
    # ...but the transmutation surface is all there.
    for present in ("Material", "PulseSchedule", "Pulse", "Cooldown", "NeutronSource"):
        assert hasattr(yani, present), f"yani should expose {present}"


def test_classes_report_their_own_package():
    """Both wheels declare the same pyclasses; each must claim its own module,
    so ``repr`` and pickling point at the package the user imported."""
    assert yamc.Material.__module__ == "yamc._core"
    assert yani.Material.__module__ == "yani._core"
