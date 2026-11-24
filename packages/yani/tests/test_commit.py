"""yani.__commit__ records the git commit the extension was built from."""
import pytest

yani = pytest.importorskip("yani", reason="standalone yani wheel not installed")


def test_build_commit_attributes():
    # Set at build time from the git checkout (see crates/yani-python/build.rs).
    # A build outside a git checkout of core reports None for both, never a
    # guess; a build inside one reports the full hash and a bool.
    commit, dirty = yani.__commit__, yani.__commit_dirty__
    if commit is None:
        assert dirty is None
    else:
        assert len(commit) == 40
        assert all(c in "0123456789abcdef" for c in commit)
        assert isinstance(dirty, bool)
