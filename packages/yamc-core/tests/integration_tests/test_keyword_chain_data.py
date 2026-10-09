"""Verify that library keywords for the transmutation subsections autoresolve
to the published release on Cloudflare R2 and assemble into a usable chain.

The four ``yamc.transmutation_*`` settings each take a library keyword or a
path. A keyword resolves the library's current release through
``{keyword}/latest.json`` and downloads
``{keyword}/{release}/transmutation/{subsection}.arrow/`` on first use, verified
against the release manifest, into ``yamc.cache_dir()``. Unset
decay/reactions/fission_yields fall back to the default ``endf-b8.1`` library.

These are end-to-end network tests against the R2 data; they skip when the
data is unavailable (offline with an empty cache, or the library not yet
published in the release layout).
"""

import os
from pathlib import Path

import pytest

import yamc


def _cache(sub, keyword="endf-b8.1"):
    # The folder the downloader writes to: the newest cached release of the
    # library, which is the one this process resolved.
    library = Path(yamc.cache_dir()) / keyword
    releases = sorted(d for d in library.glob("*") if (d / "manifest.json").is_file())
    release = releases[-1] if releases else library
    return os.path.join(release, "transmutation", f"{sub}.arrow")


def _reset():
    yamc.transmutation_decay_data = None
    yamc.transmutation_reactions = None
    yamc.transmutation_fission_yields = None
    yamc.transmutation_branch_ratios = None


def _radionuclides(nuclide_names):
    """Activation products reachable from a model whose material holds the
    given nuclides, assembling the chain from the configured subsections."""
    material = yamc.Material(
        composition={n: 1.0 for n in nuclide_names}, density=1.0, temperature=294
    )
    sphere = yamc.Sphere(radius=1.0, boundary="vacuum")
    cell = yamc.Cell(region=sphere.below, material=material)
    model = yamc.Model(
        geometry=yamc.Geometry([cell]),
        source=yamc.NeutronSource(),
    )
    return model.radionuclides()


def _try_endf_b81():
    """Trigger keyword resolution of the endf-b8.1 subsections; report success."""
    _reset()
    yamc.transmutation_decay_data = "endf-b8.1"
    yamc.transmutation_reactions = "endf-b8.1"
    yamc.transmutation_fission_yields = "endf-b8.1"
    try:
        _radionuclides(["Fe54"])
    except Exception as e:  # pragma: no cover - offline + empty cache
        # A verification or format error means the origin answered with data
        # this build refuses: a publishing mistake to fail on, not the missing
        # network or unpublished release the skip below is for.
        if "mismatch" in str(e) or "format_version" in str(e):
            raise
        return False, str(e)
    return True, ""


def test_endf_b81_subsection_keywords_resolve():
    """The endf-b8.1 keyword downloads the decay/reactions/fission_yields
    subsections from R2 and assembles a usable chain."""
    ok, err = _try_endf_b81()
    if not ok:
        pytest.skip(f"endf-b8.1 subsections unavailable (offline?): {err}")

    radio = _radionuclides(["Fe54", "Fe56", "Fe57", "Fe58"])
    assert isinstance(radio, list)
    assert len(radio) > 0, "expected at least one radionuclide for Fe isotopes"
    for sub in ("decay", "reactions", "fission_yields"):
        assert os.path.isdir(_cache(sub)), f"expected cached subsection at {_cache(sub)}"


def test_defaults_match_explicit_endf_b81():
    """Unset decay/reactions/fission_yields default to endf-b8.1, giving the
    same result as setting the keyword explicitly."""
    ok, err = _try_endf_b81()
    if not ok:
        pytest.skip(f"endf-b8.1 subsections unavailable (offline?): {err}")

    explicit = _radionuclides(["Fe54", "Fe56"])
    _reset()  # all unset -> defaults (endf-b8.1)
    default = _radionuclides(["Fe54", "Fe56"])
    assert sorted(explicit) == sorted(default)


def test_missing_subsection_errors_clearly():
    """A library that does not publish a requested subsection (tendl-2025 has
    no decay subsection) surfaces a download error, not a silent empty chain."""
    _reset()
    yamc.transmutation_decay_data = "tendl-2025"
    with pytest.raises(Exception):
        _radionuclides(["Fe54"])
    _reset()
