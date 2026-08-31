"""Guard the converted output against unintended change.

The migration off OpenMC was verified by converting nuclides with the old and new
code and diffing the results, but that was done by hand and nothing in CI would
have caught a regression afterwards. These tests convert the vendored fixtures and
compare every column against committed digests, so any change to a written value
fails here rather than waiting to be noticed downstream.

Regenerate the digests deliberately, and read the diff before accepting it:

    python tests/test_output_regression.py --update
"""

import sys
from pathlib import Path

import pytest

import arrow_digest

DATA = Path(__file__).parent / "data"
GOLDEN = DATA / "golden"


def _neutron_arrow(tmp_path, ace_path):
    from nuclear_data_to_arrow import convert_neutron
    return convert_neutron(ace_path, tmp_path, source_format="ace",
                           library="tendl-2025")


def _photon_arrow(tmp_path, photon_endf_paths):
    from nuclear_data_to_arrow import convert_photon
    photoat_path, atom_path = photon_endf_paths
    return convert_photon(photoat_path, tmp_path,
                          atom_path=atom_path,
                          library="endfb-8.1")


def test_neutron_output_is_unchanged(tmp_path, li6_ace_path):
    """Every column of the neutron output for the vendored Li6 ACE table."""
    out = _neutron_arrow(tmp_path, li6_ace_path)
    diffs = arrow_digest.compare(
        arrow_digest.read(GOLDEN / "Li6.neutron.json"),
        arrow_digest.digest_tree(out),
    )
    assert not diffs, "neutron output changed:\n  " + "\n  ".join(diffs)


def test_photon_output_is_unchanged(tmp_path, photon_endf_paths):
    """Every column of the photon output for the vendored iron evaluations."""
    out = _photon_arrow(tmp_path, photon_endf_paths)
    diffs = arrow_digest.compare(
        arrow_digest.read(GOLDEN / "Fe.photon.json"),
        arrow_digest.digest_tree(out),
    )
    assert not diffs, "photon output changed:\n  " + "\n  ".join(diffs)


def test_digest_notices_a_changed_value(tmp_path, li6_ace_path):
    """The guard has to be able to fail, so perturb one value and check it is
    reported with the column named."""
    out = _neutron_arrow(tmp_path, li6_ace_path)
    before = arrow_digest.digest_tree(out)
    after = {k: dict(v) for k, v in before.items()}
    after["reactions.arrow"]["Q_value"] = "0" * 32
    diffs = arrow_digest.compare(before, after)
    assert len(diffs) == 1
    assert "reactions.arrow" in diffs[0] and "Q_value" in diffs[0]


def test_digest_notices_a_missing_file(tmp_path, li6_ace_path):
    out = _neutron_arrow(tmp_path, li6_ace_path)
    before = arrow_digest.digest_tree(out)
    after = {k: v for k, v in before.items() if k != "fast_xs.arrow"}
    diffs = arrow_digest.compare(before, after)
    assert diffs == ["missing file: fast_xs.arrow"]


def test_digest_is_reproducible(tmp_path, li6_ace_path):
    """Converting twice in the same environment gives the same digest.

    This says nothing about reproducibility across machines, which is what the
    quantisation below is for.
    """
    a = arrow_digest.digest_tree(_neutron_arrow(tmp_path / "a", li6_ace_path))
    b = arrow_digest.digest_tree(_neutron_arrow(tmp_path / "b", li6_ace_path))
    assert arrow_digest.compare(a, b) == []


def test_digest_tolerates_last_bit_differences():
    """A one-ulp difference must not fail the guard.

    Anything computed through log, exp or an interpolating spline can move in the
    last bit between platforms, libm versions and scipy releases. In the photon
    output that covers the log-space cross sections, the bremsstrahlung electron
    energy grid and its differential cross sections. Without this tolerance the
    photon digests fail on every machine but the one that generated them, which
    is exactly what happened the first time this guard ran in CI.
    """
    for value in (1.0, 1e-30, 1.234567890123e12, -55.4544):
        nudged = value + value * 2.3e-16
        assert arrow_digest._canonical(value) == arrow_digest._canonical(nudged)


@pytest.mark.parametrize("relative", [1e-6, 1e-8, 1e-10])
def test_digest_catches_differences_worth_catching(relative):
    """The smallest real defect this guard has had to catch was a 2.6e-07
    relative error in a synthesised total, so the threshold sits well below it."""
    value = 1.2345678901234
    assert (arrow_digest._canonical(value)
            != arrow_digest._canonical(value * (1 + relative)))


def test_zero_and_negative_zero_agree():
    assert arrow_digest._canonical(0.0) == arrow_digest._canonical(-0.0)


def test_non_finite_values_are_representable():
    for value in (float("inf"), float("-inf"), float("nan")):
        assert arrow_digest._canonical(value) == repr(value)


def _update():
    """Regenerate the committed digests."""
    import gzip
    import lzma
    import tempfile
    GOLDEN.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ace = tmp / "Li6.ace"
        with gzip.open(DATA / "Li6.ace.gz", "rb") as src:
            ace.write_bytes(src.read())
        arrow_digest.write(GOLDEN / "Li6.neutron.json",
                           arrow_digest.digest_tree(_neutron_arrow(tmp / "n", ace)))
        endf_paths = []
        for name in ("photoat-026_Fe_000.endf", "atom-026_Fe_000.endf"):
            dest = tmp / name
            with lzma.open(DATA / f"{name}.xz", "rb") as src:
                dest.write_bytes(src.read())
            endf_paths.append(dest)
        arrow_digest.write(GOLDEN / "Fe.photon.json",
                           arrow_digest.digest_tree(
                               _photon_arrow(tmp / "p", tuple(endf_paths))))
    print(f"wrote digests to {GOLDEN}")


if __name__ == "__main__":
    if "--update" in sys.argv:
        _update()
    else:
        print(__doc__)
