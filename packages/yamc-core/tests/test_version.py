"""yamc.__version__ is exposed and matches the installed distribution metadata."""

from importlib.metadata import version

import yamc


def test_version_attribute_present():
    assert isinstance(yamc.__version__, str)
    assert yamc.__version__  # non-empty


def test_version_matches_distribution_metadata():
    # The single source of truth is the installed package metadata; __version__
    # must not drift from it (i.e. it is read from there, not hardcoded).
    #
    # The DISTRIBUTION is `yamc-core`; the module it provides is `yamc`, so the
    # two names differ (as `pillow` provides `PIL`). Asking for `yamc` here
    # would find the metadata-only distribution that pins this one, if it
    # happens to be installed too, and its version is a different thing.
    assert yamc.__version__ == version("yamc-core")


def test_version_not_in_public_star_export():
    # Dunder, not part of the `from yamc import *` surface; the importlib
    # helpers it is built from must not leak either.
    assert "__version__" not in yamc.__all__
    assert "PackageNotFoundError" not in yamc.__all__
