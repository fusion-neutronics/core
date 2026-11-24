"""Evaluated photonuclear processing is separate from photoatomic transport."""

import json
import lzma
from pathlib import Path

import pytest

import yamc


FIXTURES = Path(__file__).resolve().parents[3] / "crates" / "endf" / "fixtures"


@pytest.fixture
def photonuclear_input(tmp_path):
    text = lzma.decompress((FIXTURES / "n-003_Li_006_trimmed.endf.xz").read_bytes()).decode()
    lines = text.splitlines()
    header = next(index for index, line in enumerate(lines) if line[70:75] == " 1451")
    line = lines[header + 2]
    lines[header + 2] = f"{line[:44]}{0:11}{line[55:]}"
    source = tmp_path / "synthetic-photonuclear.endf"
    source.write_text("\n".join(lines))
    return source


def test_photonuclear_conversion_and_readback(photonuclear_input, tmp_path):
    folder = Path(yamc.convert_photonuclear(
        str(photonuclear_input), str(tmp_path / "photonuclear"),
        library="synthetic", data_version="2026-10-18-test",
        created_utc="2026-10-18T00:00:00Z",
    ))
    summary = yamc.read_photonuclear_from_arrow(str(folder))
    assert summary["name"] == "Li6"
    assert summary["scope_loaded"] == "evaluated"
    assert 24 in summary["mts"]
    assert (3, 24) in summary["sections"]
    assert all(count > 0 for count in summary["energy_points"].values())
    marker = json.loads((folder / "version.json").read_text())
    assert marker["particle"] == "photonuclear"
    assert marker["format_version"] == 2
    assert marker["data_version"] == "2026-10-18-test"
    assert marker["created_utc"] == "2026-10-18T00:00:00Z"
    assert not (folder / "element.arrow").exists()
    assert not (folder / "products.arrow").exists()


@pytest.mark.parametrize("filename", [
    "n-003_Li_006_trimmed.endf.xz", "photoat-001_H_000.endf.xz",
])
def test_other_sublibraries_are_rejected(filename, tmp_path):
    source = tmp_path / "wrong-sublibrary.endf"
    source.write_bytes(lzma.decompress((FIXTURES / filename).read_bytes()))
    with pytest.raises(RuntimeError, match="requires NSUB=0"):
        yamc.convert_photonuclear(str(source), str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_photonuclear_output_is_deterministic(photonuclear_input, tmp_path):
    folders = [Path(yamc.convert_photonuclear(
        str(photonuclear_input), str(tmp_path / name),
        library="synthetic", data_version="2026-10-18-test",
        created_utc="2026-10-18T00:00:00Z",
    )) for name in ("first", "second")]
    assert {path.name: path.read_bytes() for path in folders[0].iterdir()} == {
        path.name: path.read_bytes() for path in folders[1].iterdir()
    }


def test_changed_cross_section_is_rejected(photonuclear_input, tmp_path):
    ipc = pytest.importorskip("pyarrow.ipc")
    arrow = pytest.importorskip("pyarrow")
    folder = Path(yamc.convert_photonuclear(str(photonuclear_input), str(tmp_path / "out")))
    path = folder / "photonuclear.arrow"
    with ipc.open_file(path) as reader:
        schema = reader.schema
        batches = [reader.get_batch(index) for index in range(reader.num_record_batches)]
    row = batches[0].to_pylist()[0]
    row["xs"][-1] += 1.0
    batches[0] = arrow.RecordBatch.from_pylist([row], schema=schema)
    with ipc.new_file(path, schema, options=ipc.IpcWriteOptions(compression="lz4")) as writer:
        for batch in batches:
            writer.write_batch(batch)
    with pytest.raises(RuntimeError, match="xs differs from the preserved evaluation"):
        yamc.read_photonuclear_from_arrow(str(folder))