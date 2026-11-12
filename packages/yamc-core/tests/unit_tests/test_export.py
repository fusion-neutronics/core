"""Tests for ``Model.to_html(path, embed_cross_sections=...)``.

These do **not** run a simulation -- they verify the HTML the engineer
ships actually contains what each ``embed_cross_sections`` value
promises (or fails loudly when the cache doesn't have the nuclide).
A release folder under ``tmp_path`` is passed via the ``cache_dir``
kwarg. The export loads every nuclide it embeds the way the page will,
so the nuclides it embeds are copies of the repository's fixtures
(``tests/<Name>.arrow``), laid out as a native run leaves them; folders
the export only looks at the names of stay synthetic.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
from pathlib import Path

import pytest

import yamc

# Whole published sections, one folder per nuclide. The suite runs from the
# repo root (see conftest.py), where `tests` is the fixture folder.
FIXTURES = Path("tests")

# The Arrow IPC end-of-stream marker that closes a spliced stream (`EOS` in
# crates/nuclear-data-schema/src/reaction_ranges.rs).
EOS = b"\xff\xff\xff\xff\x00\x00\x00\x00"

@pytest.fixture(autouse=True)
def _explicit_export_data_sources():
    yamc.neutron_cross_section_data = "endf-b8.1"
    yamc.photon_cross_section_data = "endf-b8.1"


def test_export_requires_an_explicit_neutron_source(tmp_path):
    yamc.neutron_cross_section_data = None
    with pytest.raises(ValueError, match="neutron_cross_section_data"):
        _li6_model().to_html(tmp_path / "missing.html")
    assert not (tmp_path / "missing.html").exists()


def test_export_local_only_data_does_not_invent_a_network_library(tmp_path):
    source = tmp_path / "selected-local-data"
    source.mkdir()
    yamc.neutron_cross_section_data = str(source)
    with pytest.raises(ValueError, match="local-only"):
        _li6_model().to_html(tmp_path / "local.html")
    assert not (tmp_path / "local.html").exists()


def test_export_preserves_distinct_particle_sources(tmp_path, monkeypatch):
    from yamc import _export

    wasm = tmp_path / "metadata-only.wasm"
    glue = tmp_path / "metadata-only.js"
    wasm.write_bytes(b"\x00asm")
    glue.write_text("export default function init() {}")
    monkeypatch.setattr(_export, "_WASM_BIN", wasm)
    monkeypatch.setattr(_export, "_WASM_JS_GLUE", glue)
    yamc.neutron_cross_section_data = "fendl-3.2d"
    yamc.photon_cross_section_data = "endf-b8.1"
    base = _li6_model()
    model = yamc.Model(
        geometry=base.geometry, source=base.source, transport_secondary_photons=True,
    )
    html = model.to_html(tmp_path / "sources.html").read_text()
    match = re.search(r"const DATA_SOURCES\s*=\s*(\{[^;]*\});", html, re.DOTALL)
    assert match is not None
    sources = json.loads(match[1])
    assert sources["neutron"]["entries"]["Li6"] == "https://yamc-data.xsplot.com/fendl-3.2d"
    assert sources["photon"]["entries"]["Li"] == "https://yamc-data.xsplot.com/endf-b8.1"
    assert "fetchParticleData(JSON.stringify(DATA_SOURCES)" in html
    assert "const LIBRARY_URL = default_library_url()" not in html


def _make_fake_nuclide(cache_dir: Path, name: str, payload: bytes = b"<arrow-data>") -> None:
    """Create a fake `neutron/<name>.arrow/` directory in the release folder
    ``cache_dir`` with the standard yamc file set. Contents are arbitrary
    bytes, for nuclides the export must leave out: it never reads them."""
    nuc_dir = cache_dir / "neutron" / f"{name}.arrow"
    nuc_dir.mkdir(parents=True, exist_ok=True)
    for fname in (
        "nuclide.arrow",
        "energy.arrow",
        "reactions.arrow",
        "distributions.arrow",
        "products.arrow",
        "version.json",
    ):
        (nuc_dir / fname).write_bytes(payload + f":{name}:{fname}".encode())


def _cache_nuclide(cache_dir: Path, name: str) -> Path:
    """Copy the fixture of `name` into the release folder ``cache_dir`` as
    `neutron/<name>.arrow/`: every section whole, as a download with no
    temperature leaves it."""
    nuc_dir = cache_dir / "neutron" / f"{name}.arrow"
    shutil.copytree((FIXTURES / f"{name}.arrow").resolve(), nuc_dir)
    return nuc_dir


def _cache_nuclide_ranged(cache_dir: Path, name: str, labels: set[str]) -> Path:
    """Copy the fixture of `name` into ``cache_dir`` as a native transport run
    at the temperatures ``labels`` (``{"294K"}``) leaves it.

    That run fetches `reactions.arrow` and `energy.arrow` as the byte ranges of
    those temperatures' batches, which `version.json` indexes, splices each
    into a stream after the schema message, and caches it under
    `temperatures/` beside a record of the labels it holds. The canonical
    names stay absent. The other sections it fetches whole.
    """
    nuc_dir = _cache_nuclide(cache_dir, name)
    version = json.loads((nuc_dir / "version.json").read_text())
    indexes = {
        "reactions": [
            version["reaction_ranges"]["schema"],
            *sorted(
                span
                for by_label in version["reaction_ranges"]["mts"].values()
                for label, span in by_label.items()
                if label in labels
            ),
        ],
        "energy": [
            version["energy_ranges"]["schema"],
            *sorted(
                span
                for label, span in version["energy_ranges"]["temperatures"].items()
                if label in labels
            ),
        ],
    }
    ranged = nuc_dir / "temperatures"
    ranged.mkdir()
    for stem, spans in indexes.items():
        whole = (nuc_dir / f"{stem}.arrow").read_bytes()
        body = b"".join(whole[offset : offset + length] for offset, length, _ in spans)
        (ranged / f"{stem}.arrow").write_bytes(body + EOS)
        (ranged / f"{stem}.json").write_text(json.dumps(sorted(labels)))
        (nuc_dir / f"{stem}.arrow").unlink()
    return nuc_dir


def _li6_model(temperature: "float | None" = 294) -> "yamc.Model":
    """Minimal Li-6 sphere with one MT=105 tally."""
    li6 = yamc.Material(
        composition={"Li6": 1.0}, density=0.46, temperature=temperature, name="li6"
    )
    outer = yamc.Sphere(radius=10.0, boundary="vacuum", name="outer")
    cell = yamc.Cell(region=outer.below, material=li6, name="li6_sphere")
    geom = yamc.Geometry(cells=[cell])
    src = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1.0]),
        position=(0.0, 0.0, 0.0),
        strength=1.0,
    )
    return yamc.Model(
        geometry=geom,
        tallies=[yamc.Tally(cells=cell, scores=[105], name="trit")],
        source=src,
    )


def _extract_embedded_xs(html: str) -> dict:
    """Pull the `EMBEDDED_XS = {...};` JS literal out of an exported HTML
    and parse it back to a dict. Comments-then-assignment shape; the
    assignment line is the one starting with `const EMBEDDED_XS`."""
    m = re.search(r"const EMBEDDED_XS\s*=\s*(\{[^;]*\});", html, re.DOTALL)
    assert m is not None, "EMBEDDED_XS assignment not found in exported HTML"
    return json.loads(m.group(1))


def test_export_no_embed_writes_html_with_empty_xs(tmp_path: Path) -> None:
    """Default `embed_cross_sections=False` produces a network-dependent
    HTML -- the EMBEDDED_XS map is empty."""
    out = _li6_model().to_html(tmp_path / "out.html")
    assert out.is_file()
    html = out.read_text(encoding="utf-8")
    # Sanity: real placeholders all got substituted. A placeholder starts
    # with a letter or digit after its underscores (`__WASM_B64__`); the
    # wasm-bindgen glue's mangled closure shims contain runs of bare
    # underscores (`__closures_____invoke__`) that are not placeholders.
    leftover = re.findall(r"__[A-Z0-9][A-Z0-9_]*__", html)
    assert leftover == [], f"unsubstituted placeholders: {leftover}"
    assert _extract_embedded_xs(html) == {}


def test_export_embed_true_embeds_material_nuclides(tmp_path: Path) -> None:
    """`embed_cross_sections=True` embeds exactly the model's material
    nuclides, with every standard `.arrow` file present and decodable."""
    cache = tmp_path / "cache"
    _cache_nuclide(cache, "Li6")
    out = _li6_model().to_html(
        tmp_path / "out.html",
        embed_cross_sections=True,
        cache_dir=cache,
    )
    embedded = _extract_embedded_xs(out.read_text(encoding="utf-8"))
    assert set(embedded.keys()) == {"Li6"}
    files = embedded["Li6"]
    assert "nuclide.arrow" in files
    assert "energy.arrow" in files
    assert "reactions.arrow" in files
    # The base64 must decode cleanly back to the cached bytes.
    raw = base64.b64decode(files["nuclide.arrow"])
    assert raw == (cache / "neutron" / "Li6.arrow" / "nuclide.arrow").read_bytes()


def _load_embedded(files: dict, name: str, folder: Path, temperatures: list) -> "yamc.Nuclide":
    """Load an embedded nuclide the way the page does: its files registered
    under `/<name>.arrow/<path>` and read by the native reader at the
    temperatures of its material."""
    nuc_dir = folder / f"{name}.arrow"
    for path, data in files.items():
        (nuc_dir / path).parent.mkdir(parents=True, exist_ok=True)
        (nuc_dir / path).write_bytes(base64.b64decode(data))
    nuclide = yamc.Nuclide(name)
    nuclide.read_nuclear_data(str(nuc_dir), temperatures=temperatures)
    return nuclide


def test_embed_after_a_run_at_a_set_temperature_carries_its_ranged_sections(
    tmp_path: Path,
) -> None:
    """A transport run at a material temperature caches `reactions.arrow` and
    `energy.arrow` only as copies ranged to that temperature, under
    `temperatures/`. The export once read only the folder's own files, so it
    embedded neither, and the page said it was offline-ready, greyed out its
    fetch button and failed on Simulate with the energy grids missing."""
    cache = tmp_path / "cache"
    _cache_nuclide_ranged(cache, "Li6", {"294K"})
    out = _li6_model().to_html(
        tmp_path / "out.html", embed_cross_sections=True, cache_dir=cache
    )
    files = _extract_embedded_xs(out.read_text(encoding="utf-8"))["Li6"]
    assert set(files) == {
        "version.json",
        "nuclide.arrow",
        "products.arrow",
        "distributions.arrow",
        "temperatures/reactions.arrow",
        "temperatures/reactions.json",
        "temperatures/energy.arrow",
        "temperatures/energy.json",
    } | {name for name in files if name.endswith(".absent")}

    # What the page loads from it is what the whole sections give at 294 K.
    page = _load_embedded(files, "Li6", tmp_path / "page", ["294"])
    whole = yamc.Nuclide("Li6")
    whole.read_nuclear_data(str((FIXTURES / "Li6.arrow").resolve()), temperatures=["294"])
    assert page.loaded_temperatures == ["294"]
    assert page.reaction_mts == whole.reaction_mts
    for mt in (1, 2, 105):
        assert page.microscopic_cross_section(mt, "294") == whole.microscopic_cross_section(
            mt, "294"
        )


def test_embed_refuses_a_temperature_the_cache_does_not_hold(tmp_path: Path) -> None:
    """The ranged copy holds the temperature the native run was at. A model
    whose material is now at another one cannot run on it, and neither could
    the page, so the export says which nuclide and temperature are missing
    instead of writing that page."""
    cache = tmp_path / "cache"
    _cache_nuclide_ranged(cache, "Li6", {"294K"})
    with pytest.raises(FileNotFoundError, match=r"Cannot embed 'Li6': the page loads it at 600 K"):
        _li6_model(temperature=600).to_html(
            tmp_path / "out.html", embed_cross_sections=True, cache_dir=cache
        )
    assert not (tmp_path / "out.html").exists()


def test_embed_refuses_a_ranged_copy_for_a_material_with_no_temperature(
    tmp_path: Path,
) -> None:
    """A material with no temperature loads every temperature the data carries,
    which a copy ranged to one temperature cannot serve."""
    cache = tmp_path / "cache"
    _cache_nuclide_ranged(cache, "Li6", {"294K"})
    with pytest.raises(FileNotFoundError, match=r"the page loads it at every temperature"):
        _li6_model(temperature=None).to_html(
            tmp_path / "out.html", embed_cross_sections=True, cache_dir=cache
        )


def test_embed_refuses_a_nuclide_cached_without_its_transport_sections(
    tmp_path: Path,
) -> None:
    """An activation load caches only the cross sections it collapses (here,
    no `reactions.arrow` at all, and no products or distributions). The page
    could not run transport on that, so the export refuses it by name rather
    than embedding the rest."""
    cache = tmp_path / "cache"
    nuc_dir = _cache_nuclide(cache, "Li6")
    for section in ("reactions.arrow", "products.arrow", "distributions.arrow"):
        (nuc_dir / section).unlink()
    with pytest.raises(FileNotFoundError, match=r"Cannot embed 'Li6'.*reactions\.arrow"):
        _li6_model().to_html(tmp_path / "out.html", embed_cross_sections=True, cache_dir=cache)


def test_export_embed_element_expands_to_all_isotopes(tmp_path: Path) -> None:
    """Element symbols expand to every `Element<digits>` directory present
    in the cache. Lets the engineer pre-bake related isotopes the
    recipient might switch to."""
    cache = tmp_path / "cache"
    _cache_nuclide(cache, "Li6")
    _cache_nuclide(cache, "Li7")
    # Decoy: same-prefix nuclide that ISN'T an isotope of Li -- must not match.
    _make_fake_nuclide(cache, "Be9")
    # Decoy: bare-element folder `neutron/Li.arrow` -- must not match (we
    # only match `Li<digits>`, not the natural-element entry).
    (cache / "neutron" / "Li.arrow").mkdir()
    (cache / "neutron" / "Li.arrow" / "nuclide.arrow").write_bytes(b"natural")

    out = _li6_model().to_html(
        tmp_path / "out.html",
        embed_cross_sections=["Li"],
        cache_dir=cache,
    )
    embedded = _extract_embedded_xs(out.read_text(encoding="utf-8"))
    assert set(embedded.keys()) == {"Li6", "Li7"}


def test_export_embed_explicit_nuclide_keeps_unrelated_out(tmp_path: Path) -> None:
    """An explicit nuclide list embeds exactly those, even if the cache
    has more available -- Fe56 not in the list shouldn't be embedded."""
    cache = tmp_path / "cache"
    _cache_nuclide(cache, "Li6")
    _make_fake_nuclide(cache, "Fe56")
    out = _li6_model().to_html(
        tmp_path / "out.html",
        embed_cross_sections=["Li6"],
        cache_dir=cache,
    )
    embedded = _extract_embedded_xs(out.read_text(encoding="utf-8"))
    assert set(embedded.keys()) == {"Li6"}


def test_export_missing_nuclide_raises_with_helpful_message(tmp_path: Path) -> None:
    """When a requested nuclide isn't in the cache, the error names the
    expected path so the engineer knows what to do."""
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        _li6_model().to_html(
            tmp_path / "out.html",
            embed_cross_sections=["Cm245"],
            cache_dir=cache,
        )
    msg = str(excinfo.value)
    assert "Cm245" in msg
    assert "Cm245.arrow" in msg


def test_export_unknown_element_raises_with_helpful_message(tmp_path: Path) -> None:
    """Element symbol with no matching isotopes in the cache is an error."""
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        _li6_model().to_html(
            tmp_path / "out.html",
            embed_cross_sections=["Zz"],
            cache_dir=cache,
        )
    assert "Zz" in str(excinfo.value)


def test_export_invalid_embed_entry_raises_value_error(tmp_path: Path) -> None:
    """Non-string entries are caught early with a clear ValueError."""
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(ValueError, match="non-empty strings"):
        _li6_model().to_html(
            tmp_path / "out.html",
            embed_cross_sections=[123],  # type: ignore[list-item]
            cache_dir=cache,
        )


def test_exported_html_contains_load_model_json_and_plot(tmp_path: Path) -> None:
    """Sanity: the exported HTML wires the wasm load + the iframe-mounted
    plot, so the rest of the page can run."""
    html = _li6_model().to_html(tmp_path / "out.html").read_text(encoding="utf-8")
    assert "sim.load_model_json(MODEL_JSON)" in html
    # Plot iframe + the JS that drives its refresh-on-Apply.
    assert 'id="plot-iframe"' in html
    assert "sim.plotHtml" in html
    assert "PLOT.redraw" in html
    # The wasm base64 payload should be substantial (the binary is >1 MB).
    m = re.search(r'const WASM_B64 = "([A-Za-z0-9+/=]+)"', html)
    assert m is not None and len(m.group(1)) > 1_000_000


def test_exported_html_fetches_through_the_wasm_fetcher(tmp_path: Path) -> None:
    """The page's Fetch button goes through `WasmSimulation.fetchParticleData`,
    which reads the published per-section objects. The page once fetched
    `.arrow.tar` bundles the CDN had stopped serving, and nothing here
    noticed; this pins the wiring, and the browser test in
    crates/yamc/tests/wasm_fetch_browser.rs exercises the fetch itself."""
    html = _li6_model().to_html(tmp_path / "out.html").read_text(encoding="utf-8")
    assert "sim.fetchParticleData(" in html
    assert "const LIBRARY_URL = default_library_url()" not in html
    assert "sim.model_missing_nuclides()" in html
    # The greyed-out Simulate button says what to do, before the fetch and
    # again whenever an edit adds a nuclide the store lacks.
    assert 'id="sim" disabled>Fetch the cross section data first<' in html
    assert '"Fetch the cross section data first" : "Simulate"' in html
    assert ".arrow.tar" not in html
    assert "parseTar" not in html


def test_editor_sections_are_collapsible_and_closed_by_default(tmp_path: Path) -> None:
    """Materials / surfaces / tallies sections render as <details>
    (native disclosure widget) so they collapse to one line by default
    and the recipient expands the ones they care about. Locks the
    pattern in so a future template refactor doesn't accidentally
    revert to always-expanded fieldsets."""
    html = _li6_model().to_html(tmp_path / "out.html").read_text(encoding="utf-8")
    for section in ("surfaces", "materials", "tallies"):
        m = re.search(rf'<details id="{section}-fieldset"[^>]*>', html)
        assert m is not None, f"<details> for {section} missing"
        tag = m.group(0)
        # No `open` attribute → starts collapsed.
        assert " open" not in tag, f"{section} should start collapsed, got: {tag}"


def test_exported_html_has_fullscreen_buttons_on_both_plots(tmp_path: Path) -> None:
    """Both the geometry and tally-plot iframes get a "⛶ Full screen"
    button. The iframes carry `allow="fullscreen"` so the click-driven
    `requestFullscreen()` is permitted. Locks the pattern in so future
    template edits don't silently drop it."""
    html = _li6_model().to_html(tmp_path / "out.html").read_text(encoding="utf-8")

    # Buttons exist, each targeting its iframe by data-target.
    assert 'class="fullscreen-btn" data-target="plot-iframe"' in html
    assert 'class="fullscreen-btn" data-target="tally-plot-iframe"' in html

    # Both iframes allow fullscreen (browsers refuse requestFullscreen
    # otherwise).
    assert html.count('allow="fullscreen"') == 2

    # The delegated click handler is present and wires
    # `requestFullscreen` against the data-target id.
    assert "fullscreen-btn" in html
    assert "requestFullscreen" in html


def _extract_section_flags(html: str) -> dict:
    m = re.search(r"const SECTION_FLAGS\s*=\s*(\{[^;]*\});", html, re.DOTALL)
    assert m is not None, "SECTION_FLAGS not found in exported HTML"
    return json.loads(m.group(1))


def test_export_default_flags_all_sections_visible(tmp_path: Path) -> None:
    """No flags passed → every editor section ships."""
    html = _li6_model().to_html(tmp_path / "out.html").read_text(encoding="utf-8")
    assert _extract_section_flags(html) == {
        "geometry": True, "surfaces": True, "materials": True, "tallies": True,
    }
    # And the fieldsets themselves are present in the template.
    for section in ("geometry", "surfaces", "materials", "tallies"):
        assert f'data-section="{section}"' in html


def test_export_show_flags_round_trip(tmp_path: Path) -> None:
    """Each show_* kwarg toggles its flag in the embedded SECTION_FLAGS."""
    html = _li6_model().to_html(
        tmp_path / "out.html",
        show_geometry=False,
        show_surfaces=False,
        show_materials=True,
        show_tallies=False,
    ).read_text(encoding="utf-8")
    assert _extract_section_flags(html) == {
        "geometry": False, "surfaces": False, "materials": True, "tallies": False,
    }


def test_export_surfaces_appears_before_materials(tmp_path: Path) -> None:
    """The HTML template orders surfaces above materials so the recipient
    sees the riskier (geometry-altering) edits first."""
    html = _li6_model().to_html(tmp_path / "out.html").read_text(encoding="utf-8")
    surf_idx = html.index('data-section="surfaces"')
    mat_idx = html.index('data-section="materials"')
    assert surf_idx < mat_idx, "surfaces fieldset must come before materials"
