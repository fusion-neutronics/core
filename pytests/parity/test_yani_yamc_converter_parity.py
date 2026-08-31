"""The two wheels must write the same data.

``yamc`` and ``yani`` ship the same converters. They are one Rust
implementation registered into two extension modules, so today they cannot
disagree, and that is exactly what makes a test worth having: nothing else
would catch a divergence introduced later. A second code path added "just for
yamc" would produce data that silently depends on which wheel generated it.

What is compared is every field of every column, exactly, plus the schema and
its metadata. Not a tolerance: two conversions that agree to 1e-12 are not the
same data, and the published libraries are distributed as files.

The comparison is of the DATA rather than the raw bytes, and both now hold:
the schema metadata is built to iterate in sorted key order (issue #441), so
two conversions of the same input are byte-identical and so are the two
wheels. `test_output_is_byte_reproducible` below asserts the bytes; these
compare the data, so that a failure says which column differs instead of
"these 232282 bytes are not those 232282 bytes".

The split of responsibilities the tests below pin down:

* transmutation, branching and the neutron routes are on both wheels, because
  ``pip install yani`` has to be able to produce a transmutation chain without
  dragging in the transport stack;
* photon conversion is on both, but only ``yamc`` ships the auxiliary Compton,
  bremsstrahlung and density-effect tabulations, because they are not in any
  evaluation and transmutation never needs them. ``yamc.convert_photon``
  therefore defaults to its own copies where ``yani.convert_photon`` must be
  given the three paths.
"""

import lzma
from pathlib import Path

import pytest

import yamc

yani = pytest.importorskip(
    "yani",
    reason="the parity test needs both wheels installed in one environment",
)

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "crates" / "endf" / "fixtures"
PHOTON_DATA = REPO / "packages" / "yamc-core" / "python" / "yamc" / "data"


def _unpack(name, dest):
    """Decompress an xz fixture, since the converters read plain files."""
    src = FIXTURES / name
    if not src.is_file():
        pytest.skip(f"fixture {name} is absent")
    dest.write_bytes(lzma.decompress(src.read_bytes()))
    return dest


def _sections(root):
    """Every file under a directory, by relative path."""
    return {
        str(p.relative_to(root)): p
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _read(path):
    """An Arrow file as (schema, metadata, every column of every batch).

    Read through pyarrow rather than compared as bytes, so the comparison is of
    the data and not of `HashMap` iteration order. `reactions.arrow` is written
    one record batch per row, so every batch is read rather than just the
    first.
    """
    # Skipped rather than errored when pyarrow is absent, matching the two
    # wheels above. The parity CI job installs it, and the conftest's strict
    # mode refuses to run there without it, so this never skips where it counts.
    ipc = pytest.importorskip("pyarrow.ipc")

    if path.suffix != ".arrow":
        return ("raw", path.read_bytes())
    with ipc.open_file(path) as reader:
        schema = reader.schema
        batches = [reader.get_batch(i) for i in range(reader.num_record_batches)]
    return (
        [(f.name, str(f.type), f.nullable) for f in schema],
        dict(schema.metadata or {}),
        [[b.column(i).to_pylist() for i in range(b.num_columns)] for b in batches],
    )


def _assert_identical(a, b, what):
    """Both trees must hold the same files with the same contents."""
    left, right = _sections(a), _sections(b)
    assert set(left) == set(right), (
        f"{what}: the two wheels wrote different files. "
        f"yamc only: {sorted(set(left) - set(right))}, "
        f"yani only: {sorted(set(right) - set(left))}"
    )
    assert left, f"{what}: nothing was written, so this test proved nothing"
    for name in sorted(left):
        assert _read(left[name]) == _read(right[name]), (
            f"{what}: {name} differs between the wheels"
        )
    return sorted(left)


STAMP = "2026-01-01T00:00:00+00:00"


def test_neutron_cross_sections_are_identical(tmp_path):
    """The ACE route, which needs no NJOY and so runs anywhere."""
    ace = _unpack("Li6.ace.xz", tmp_path / "Li6.ace")
    out_yamc = tmp_path / "yamc"
    out_yani = tmp_path / "yani"

    kwargs = dict(
        source_format="ace",
        library="endf-b8.1",
        data_version="parity",
        created_utc=STAMP,
    )
    yamc.convert_neutron_xs(input_path=str(ace), output_dir=str(out_yamc), **kwargs)
    yani.convert_neutron_xs(input_path=str(ace), output_dir=str(out_yani), **kwargs)

    files = _assert_identical(out_yamc, out_yani, "neutron cross sections")
    assert any(f.endswith("reactions.arrow") for f in files)


def test_transmutation_chain_is_identical(tmp_path):
    """The chain, built from the decay fixtures both wheels carry."""
    decay = []
    for name in ("dec-055_Cs_137.endf.xz", "dec-054_Xe_136.endf.xz"):
        decay.append(str(_unpack(name, tmp_path / name.replace(".xz", ""))))

    out_yamc = tmp_path / "chain-yamc.arrow"
    out_yani = tmp_path / "chain-yani.arrow"
    kwargs = dict(
        decay_files=decay,
        fpy_files=[],
        neutron_files=[],
        library="endf-b8.1",
        data_version="parity",
        created_utc=STAMP,
        subsections=["decay"],
    )
    yamc.convert_transmutation(output_path=str(out_yamc), **kwargs)
    yani.convert_transmutation(output_path=str(out_yani), **kwargs)

    files = _assert_identical(out_yamc, out_yani, "transmutation chain")
    assert any("decay" in f for f in files)


def test_photon_is_identical_when_yani_is_given_the_tabulations(tmp_path):
    """Photon conversion agrees, and shows why the default lives on yamc.

    ``yamc.convert_photon`` finds the three auxiliary tabulations in its own
    package. ``yani`` does not ship them, so the same call has to be handed
    their paths. Given the same inputs the two produce the same data; the
    difference between the wheels is what they can find, not what they compute.
    """
    photoatomic = _unpack("photoat-001_H_000.endf.xz", tmp_path / "photoat-H.endf")

    tabulations = {
        "compton_profiles": str(PHOTON_DATA / "compton_profiles_biggs1975.txt"),
        "density_effect": str(PHOTON_DATA / "density_effect_sternheimer1982.txt"),
        "bremsstrahlung": str(PHOTON_DATA / "bremsstrahlung_seltzer_berger1986.txt"),
    }
    for path in tabulations.values():
        assert Path(path).is_file(), f"the bundled tabulation {path} is missing"

    out_yamc = tmp_path / "ph-yamc"
    out_yani = tmp_path / "ph-yani"
    # yamc: no tabulation paths, it uses the ones it ships.
    yamc.convert_photon(
        photoatomic_path=str(photoatomic),
        output_dir=str(out_yamc),
        library="endf-b8.1",
        data_version="parity",
        created_utc=STAMP,
    )
    # yani: the same files, named explicitly.
    yani.convert_photon(
        photoatomic_path=str(photoatomic),
        output_dir=str(out_yani),
        library="endf-b8.1",
        data_version="parity",
        created_utc=STAMP,
        **tabulations,
    )

    files = _assert_identical(out_yamc, out_yani, "photon sections")
    assert any(f.endswith("element.arrow") for f in files)


def test_both_wheels_expose_the_same_converters():
    """Neither wheel may quietly lose a converter the other has.

    A missing function would make a generation script work against one wheel
    and fail against the other, which is the failure this whole file exists to
    prevent, one level up from the bytes.
    """
    expected = {
        "convert_transmutation",
        "convert_branching",
        "convert_neutron_xs",
        "convert_neutron_transport",
        "convert_photon",
    }
    missing_yamc = {n for n in expected if not callable(getattr(yamc, n, None))}
    missing_yani = {n for n in expected if not callable(getattr(yani, n, None))}
    assert not missing_yamc, f"yamc is missing {sorted(missing_yamc)}"
    assert not missing_yani, f"yani is missing {sorted(missing_yani)}"


def test_yamc_refuses_a_partial_set_of_tabulations(tmp_path):
    """Two of the three is a mistake, not a request for a mixed set.

    The three tabulations are read together. Accepting a partial set would
    silently drop `compton.arrow` or `bremsstrahlung.arrow` from the output,
    which loads fine and is missing physics.
    """
    photoatomic = _unpack("photoat-001_H_000.endf.xz", tmp_path / "photoat-H.endf")
    with pytest.raises(ValueError, match="all three or none"):
        yamc.convert_photon(
            photoatomic_path=str(photoatomic),
            output_dir=str(tmp_path / "out"),
            compton_profiles=str(PHOTON_DATA / "compton_profiles_biggs1975.txt"),
        )


def test_output_is_byte_reproducible(tmp_path):
    """The same input, converted repeatedly, must give the same bytes.

    Not a nicety. A published library that cannot be checksummed cannot be
    verified by whoever mirrors it, and a rebuild that changes every file makes
    "did anything actually change?" unanswerable by comparison. Both questions
    came up while deciding which nuclides needed republishing.

    This used to fail. Arrow serialises schema metadata in map iteration order
    and `std`'s `HashMap` seeds per instance, so `element.arrow` came out as one
    of two byte patterns differing in 114 bytes (issue #441). Twelve runs is
    enough to catch a two-way flip with probability 1 - 2^-11.
    """
    import hashlib

    photoatomic = _unpack("photoat-001_H_000.endf.xz", tmp_path / "photoat-H.endf")
    digests = {}
    for i in range(12):
        out = tmp_path / f"run{i}"
        yamc.convert_photon(
            photoatomic_path=str(photoatomic),
            output_dir=str(out),
            library="endf-b8.1",
            data_version="repro",
            created_utc=STAMP,
        )
        for written in sorted((out / "H.arrow").iterdir()):
            digests.setdefault(written.name, set()).add(
                hashlib.sha256(written.read_bytes()).hexdigest()
            )

    assert digests, "nothing was written, so this test proved nothing"
    unstable = {name: len(d) for name, d in digests.items() if len(d) != 1}
    assert not unstable, (
        f"these sections differ between identical runs: {unstable}. "
        f"Schema metadata is probably no longer built in sorted key order."
    )
