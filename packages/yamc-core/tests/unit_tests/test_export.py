"""Tests for ``Model.to_html(path, embed_cross_sections=...)``.

These do **not** run a simulation -- they verify the HTML the engineer
ships actually contains what each ``embed_cross_sections`` value
promises (or fails loudly when the cache doesn't have the nuclide).
A hermetic fixture cache (synthetic `.arrow` directories under
``tmp_path``) is passed via the ``cache_dir`` kwarg so the test
doesn't depend on the developer machine having real ENDF data
downloaded.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import pytest

import yamc


def _make_fake_nuclide(cache_dir: Path, name: str, payload: bytes = b"<arrow-data>") -> None:
    """Create a fake `endf-b8.1-<name>.arrow/` directory in ``cache_dir`` with
    the standard yamc file set. Contents are arbitrary bytes -- the export path
    doesn't parse them, only base64-encodes."""
    nuc_dir = cache_dir / f"endf-b8.1-{name}.arrow"
    nuc_dir.mkdir(parents=True, exist_ok=True)
    for fname in (
        "nuclide.arrow",
        "reactions.arrow",
        "distributions.arrow",
        "products.arrow",
        "fast_xs.arrow",
        "version.json",
    ):
        (nuc_dir / fname).write_bytes(payload + f":{name}:{fname}".encode())


def _li6_model() -> "yamc.Model":
    """Minimal Li-6 sphere with one MT=105 tally."""
    li6 = yamc.Material(
        composition={"Li6": 1.0}, density=0.46, temperature=294, name="li6"
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
    # Sanity: real placeholders all got substituted.
    leftover = re.findall(r"__[A-Z0-9_]+__", html)
    assert leftover == [], f"unsubstituted placeholders: {leftover}"
    assert _extract_embedded_xs(html) == {}


def test_export_embed_true_embeds_material_nuclides(tmp_path: Path) -> None:
    """`embed_cross_sections=True` embeds exactly the model's material
    nuclides, with every standard `.arrow` file present and decodable."""
    cache = tmp_path / "cache"
    _make_fake_nuclide(cache, "Li6")
    out = _li6_model().to_html(
        tmp_path / "out.html",
        embed_cross_sections=True,
        cache_dir=cache,
    )
    embedded = _extract_embedded_xs(out.read_text(encoding="utf-8"))
    assert set(embedded.keys()) == {"Li6"}
    files = embedded["Li6"]
    assert "nuclide.arrow" in files
    assert "reactions.arrow" in files
    # The base64 must decode cleanly back to non-empty bytes.
    raw = base64.b64decode(files["nuclide.arrow"])
    assert raw.startswith(b"<arrow-data>")


def test_export_embed_element_expands_to_all_isotopes(tmp_path: Path) -> None:
    """Element symbols expand to every `Element<digits>` directory present
    in the cache. Lets the engineer pre-bake related isotopes the
    recipient might switch to."""
    cache = tmp_path / "cache"
    _make_fake_nuclide(cache, "Li6")
    _make_fake_nuclide(cache, "Li7")
    # Decoy: same-prefix nuclide that ISN'T an isotope of Li -- must not match.
    _make_fake_nuclide(cache, "Be9")
    # Decoy: bare-element file `endf-b8.1-Li.arrow` -- must not match (we
    # only match `Li<digits>`, not the natural-element entry).
    (cache / "endf-b8.1-Li.arrow").mkdir()
    (cache / "endf-b8.1-Li.arrow" / "nuclide.arrow").write_bytes(b"natural")

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
    _make_fake_nuclide(cache, "Li6")
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
    assert "endf-b8.1-Cm245.arrow" in msg


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
