"""Choosing which NJOY executable a conversion runs.

Which NJOY is used is not a preference. FENDL is processed by the IAEA-NDS fork
of NJOY2016, and against upstream NJOY the URR probability tables differ by up
to 100% (La139) and MT 301 heating / MT 444 damage by up to 77% below roughly
1 keV. Both builds succeed, so the wrong choice is quiet.

This package is a library with no command line of its own, so the option is a
keyword argument. The ``--njoy-exec`` flag that surfaces it, and the tests for
the argument parsing, live with the drivers in nuclear_data_generation_scripts.
"""

import pytest


def test_convert_neutron_forwards_njoy_exec(monkeypatch, tmp_path):
    """``njoy_exec`` must reach endf-python's from_njoy, not be swallowed."""
    import endf

    import nuclear_data_to_arrow as nda

    seen = {}

    class _FakeNeutron:
        name = "Fe56"

    def fake_from_njoy(path, **kwargs):
        seen.update(kwargs)
        return _FakeNeutron()

    monkeypatch.setattr(endf.IncidentNeutron, "from_njoy",
                        staticmethod(fake_from_njoy))
    monkeypatch.setattr(nda, "export_neutron_to_arrow",
                        lambda data, path, library="", data_version="": None)

    nda.convert_neutron("Fe56.endf", tmp_path, source_format="endf",
                        njoy_exec="/opt/iaea/njoy")

    assert seen["njoy_exec"] == "/opt/iaea/njoy"


def test_convert_neutron_defaults_njoy_exec(monkeypatch, tmp_path):
    """Unset, it resolves ``njoy`` from PATH."""
    import endf

    import nuclear_data_to_arrow as nda

    seen = {}

    class _FakeNeutron:
        name = "Fe56"

    monkeypatch.setattr(
        endf.IncidentNeutron, "from_njoy",
        staticmethod(lambda path, **kw: (seen.update(kw), _FakeNeutron())[1]))
    monkeypatch.setattr(nda, "export_neutron_to_arrow",
                        lambda data, path, library="", data_version="": None)

    nda.convert_neutron("Fe56.endf", tmp_path, source_format="endf")

    assert seen["njoy_exec"] == "njoy"


def test_ace_route_never_runs_njoy(monkeypatch, tmp_path):
    """The ACE route must not pass njoy_exec anywhere: it runs no NJOY."""
    import endf

    import nuclear_data_to_arrow as nda

    class _FakeNeutron:
        name = "Li6"

    called = []
    monkeypatch.setattr(endf.IncidentNeutron, "from_njoy",
                        staticmethod(lambda *a, **kw: called.append(kw)))
    monkeypatch.setattr(endf.IncidentNeutron, "from_ace",
                        staticmethod(lambda path: _FakeNeutron()))
    monkeypatch.setattr(nda, "export_neutron_to_arrow",
                        lambda data, path, library="", data_version="": None)

    nda.convert_neutron("Li6.ace", tmp_path, source_format="ace",
                        njoy_exec="/opt/iaea/njoy")

    assert called == []


def test_package_installs_no_console_scripts():
    """The converter is a library. Every entry point lives with the drivers.

    Guards against a [project.scripts] block creeping back in: the pipeline had
    two invocation conventions at once (console scripts here, ``python -m``
    there) and settling on one was deliberate.
    """
    # importlib.metadata rather than tomllib, which is 3.11+ and would fail on
    # this package's own 3.10 floor. It also checks the installed distribution
    # rather than the source file, which is the thing that actually matters.
    from importlib.metadata import distribution

    entry_points = distribution("nuclear_data_to_arrow").entry_points
    scripts = sorted(ep.name for ep in entry_points if ep.group == "console_scripts")
    assert not scripts, f"console scripts reintroduced: {scripts}"


def test_no_cli_module_remains():
    with pytest.raises(ModuleNotFoundError):
        __import__("nuclear_data_to_arrow.cli")
