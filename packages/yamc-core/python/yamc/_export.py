"""Implementation of ``Model.to_html(path, ...)``.

Bundles the model JSON, the wasm transport binary, the wasm-bindgen JS
glue, a live JS-rendered geometry+source plot, and (optionally) nuclide
cross-section data into a single self-contained HTML file. The recipient
opens the HTML and runs simulations in-browser -- no Python install, no
admin rights, no executable permissions, no online connection if the
required nuclide data was embedded. The engineer can also lock down
which editor sections the recipient sees via the ``show_*`` kwargs.

Wired onto ``Model`` as ``Model.to_html = to_html`` in
``yamc/__init__.py``.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Iterable

# Bundle locations relative to this module -- these all ship inside the
# installed wheel via maturin's `python-source = "python"`.
_HERE = Path(__file__).parent
_TEMPLATE_DIR = _HERE / "_browser_template"
_HTML_TEMPLATE = _TEMPLATE_DIR / "index.html"
_APP_JS = _TEMPLATE_DIR / "app.js"
_WASM_DIR = _HERE / "_wasm"
_WASM_BIN = _WASM_DIR / "yamc_sim_bg.wasm"
_WASM_JS_GLUE = _WASM_DIR / "yamc_sim.js"

# Local on-disk cross-section cache yamc populates via the URL cache layer.
# Same layout the wasm side expects after extraction:
#   ~/.cache/yamc/endf-b8.1-<Name>.arrow/{nuclide,reactions,distributions,...}.arrow
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "yamc"
_NUCLIDE_DIR_PREFIX = "endf-b8.1-"
_NUCLIDE_DIR_SUFFIX = ".arrow"


def _expand_nuclide_or_element(item: str, cache_dir: Path) -> list[str]:
    """Resolve a single user-supplied entry to a list of nuclide names.

    - ``"Li6"`` → ``["Li6"]`` (already a nuclide)
    - ``"Li"``  → ``["Li6", "Li7"]`` (element → all isotopes available
      in the local cache)

    Heuristic: anything containing a digit is treated as a nuclide,
    otherwise as an element to expand. Element expansion is strict --
    matches ``Element + one-or-more-digits``, so ``Li`` won't match the
    natural-element file ``endf-b8.1-Li.arrow``.
    """
    if re.search(r"\d", item):
        return [item]
    iso_re = re.compile(rf"^{re.escape(item)}(\d+)$")
    expansions: list[str] = []
    if cache_dir.is_dir():
        for entry in sorted(cache_dir.iterdir()):
            name = entry.name
            if not name.startswith(_NUCLIDE_DIR_PREFIX) or not name.endswith(_NUCLIDE_DIR_SUFFIX):
                continue
            short = name[len(_NUCLIDE_DIR_PREFIX) : -len(_NUCLIDE_DIR_SUFFIX)]
            if iso_re.match(short):
                expansions.append(short)
    if not expansions:
        raise FileNotFoundError(
            f"Could not expand element '{item}' -- no isotope files matching "
            f"{cache_dir}/{_NUCLIDE_DIR_PREFIX}{item}<digits>{_NUCLIDE_DIR_SUFFIX}/. "
            f"Either pass explicit nuclide names (e.g. ['{item}1']) or run "
            f"yamc once with this element in a material so the cache populates."
        )
    return expansions


def _resolve_embed_list(
    embed: "bool | Iterable[str]",
    required: list[str],
    cache_dir: Path,
) -> list[str]:
    """Translate the ``embed_cross_sections`` arg into a concrete nuclide list."""
    if embed is False or embed is None:
        return []
    if embed is True:
        # Embed exactly what the current materials need.
        return list(required)
    if isinstance(embed, str):
        # Probably a user mistake -- accept a single string as a one-element list.
        embed = [embed]
    resolved: list[str] = []
    seen: set[str] = set()
    for item in embed:
        if not isinstance(item, str) or not item:
            raise ValueError(f"embed_cross_sections entries must be non-empty strings, got {item!r}")
        for nuc in _expand_nuclide_or_element(item, cache_dir):
            if nuc not in seen:
                seen.add(nuc)
                resolved.append(nuc)
    return resolved


def _load_nuclide_files(name: str, cache_dir: Path) -> dict[str, bytes]:
    """Read every `.arrow` file for a single nuclide from the local cache.

    Returns {filename → raw bytes}. The wasm side will register each as
    ``/<name>.arrow/<filename>``.
    """
    nuc_dir = cache_dir / f"{_NUCLIDE_DIR_PREFIX}{name}{_NUCLIDE_DIR_SUFFIX}"
    if not nuc_dir.is_dir():
        raise FileNotFoundError(
            f"Nuclide '{name}' is not in the local cache at {nuc_dir}. "
            f"Either embed only nuclides you already have, or run a yamc "
            f"simulation that references this nuclide (so its data downloads), "
            f"then re-export."
        )
    files: dict[str, bytes] = {}
    for entry in sorted(nuc_dir.iterdir()):
        if entry.is_file():
            files[entry.name] = entry.read_bytes()
    if not files:
        raise FileNotFoundError(f"Nuclide directory {nuc_dir} is empty")
    return files


def _build_embedded_xs_json(nuclides: list[str], cache_dir: Path) -> str:
    """JSON payload for the `EMBEDDED_XS` global.

    Shape: ``{nuclide: {filename: base64_bytes, ...}, ...}``.
    """
    payload: dict[str, dict[str, str]] = {}
    for name in nuclides:
        files = _load_nuclide_files(name, cache_dir)
        payload[name] = {
            filename: base64.b64encode(data).decode("ascii")
            for filename, data in files.items()
        }
    return json.dumps(payload)


def to_html(
    self,
    path: "str | Path",
    embed_cross_sections: "bool | Iterable[str]" = False,
    *,
    show_geometry: bool = True,
    show_surfaces: bool = True,
    show_materials: bool = True,
    show_tallies: bool = True,
    tally_plot_html: "str | None" = None,
    initial_result_text: "str | None" = None,
    cache_dir: "str | Path | None" = None,
) -> Path:
    """Export this model to a self-contained, browser-runnable HTML file.

    Args:
        path: Output HTML path.
        embed_cross_sections: Controls the in-HTML nuclide data:

            - ``False`` (default): no XS data embedded. The recipient
              must be online and the page fetches from yamc-data.xsplot.com.
            - ``True``: embed exactly the nuclides currently referenced by
              the model's materials. Makes the HTML fully offline-runnable
              *until* the recipient edits the materials to add new ones.
            - list of ``str``: explicit nuclides or elements. Each entry is
              either a nuclide name (``"Li6"``, kept as-is) or an element
              symbol (``"Li"``, expanded to every ``Li<digits>`` directory in
              the local cache). Use this to anticipate the recipient swapping
              in a related isotope without needing network access.

        show_geometry: When False, hides the geometry + source plot.
        show_surfaces: When False, hides the surfaces editor -- the
            recipient cannot change radii / plane offsets / etc.
        show_materials: When False, hides the materials editor -- the
            recipient cannot change composition or density.
        show_tallies: When False, hides the tally editor -- the recipient
            cannot add or remove scores.
        tally_plot_html: Optional HTML fragment to embed below the
            Simulate result panel as a static snapshot of an interactive
            tally plot. Pass ``tally.plot(...)._repr_html_()`` from a
            mesh-tally simulation you ran in Python before exporting --
            the recipient sees the spatial distribution baked in. It
            does **not** refresh on in-browser re-simulate (no wasm
            tally plotter yet); the caption tells the user that.
        initial_result_text: Optional pre-run scalar tally text to seed
            the result panel with (otherwise it shows "No result yet").
            Useful alongside ``tally_plot_html`` so the recipient sees a
            consistent set of pre-run numbers next to the spatial plot
            on first load. The Simulate handler overwrites it with the
            recipient's own result after they click.
        cache_dir: Override the local on-disk XS cache (default:
            ``~/.cache/yamc``). Useful for tests or vendored data layouts.

    Returns:
        The output path (as a :class:`~pathlib.Path`).

    Notes:
        Embedded data inflates the HTML size by ~the raw byte count × 1.33
        for base64. ENDF/B-VIII.1 nuclides range from ~2 MB (Li-6) to
        ~150 MB (Fe-56). Embedding everything in a heavy-element material
        produces a multi-hundred-MB HTML; choose ``embed_cross_sections``
        accordingly.

        The ``show_*`` flags are presentation-only -- the recipient can
        still inspect (or open devtools and edit) the embedded model
        JSON. Use them to declutter the UI, not as a security boundary.

    Example::

        >>> model.to_html("portable.html", embed_cross_sections=True)
        >>> # The HTML now runs without internet access.

        >>> # Anticipate the recipient swapping Li6 for Li7:
        >>> model.to_html("portable.html", embed_cross_sections=["Li"])

        >>> # Materials-only demo -- recipient can change density / composition
        >>> # but cannot touch the geometry or the tally definitions.
        >>> model.to_html("locked.html", show_surfaces=False, show_tallies=False)
    """
    out = Path(path)
    cache = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR

    required = self.required_nuclides()
    embed_list = _resolve_embed_list(embed_cross_sections, required, cache)

    embedded_xs_json = _build_embedded_xs_json(embed_list, cache) if embed_list else "{}"

    # Photon transport needs per-element data too. When the engineer asked
    # to embed (any truthy `embed_cross_sections`) and the model uses photon
    # transport, embed every element the materials reference so the offline
    # HTML can run photon transport with no network. `_build_embedded_xs_json`
    # reads `endf-b8.1-<El>.arrow/` dirs (element.arrow, subshells.arrow, ...)
    # the same way it reads neutron nuclide dirs.
    photon_elements = self.required_elements() if embed_list else []
    embedded_photon_json = (
        _build_embedded_xs_json(photon_elements, cache) if photon_elements else "{}"
    )

    # Section visibility flags -- app.js hides fieldsets whose flag is false.
    # Keys match the dataset attributes / fieldset ids in index.html.
    section_flags_json = json.dumps(
        {
            "geometry": bool(show_geometry),
            "surfaces": bool(show_surfaces),
            "materials": bool(show_materials),
            "tallies": bool(show_tallies),
        }
    )

    model_json = self.to_json()
    wasm_b64 = base64.b64encode(_WASM_BIN.read_bytes()).decode("ascii")
    js_glue_src = _WASM_JS_GLUE.read_text(encoding="utf-8")

    # The geometry+source plot is rendered client-side in JS from the
    # model JSON (see app.js `PLOT` block), so the engineer's exported
    # model and the recipient's edited model both display correctly
    # without a Python round-trip. `Model.plot()` is *not* called here.

    # Substitution order matters: `__APP_JS__` brings in inner
    # placeholders (__WASM_B64__, __JS_SRC_JSON__, __MODEL_JSON__,
    # __EMBEDDED_XS__, __SECTION_FLAGS__) that later passes must see.
    # Do it first.
    # Static tally-plot snapshot: passed as a complete HTML document
    # (from `tally.plot()._repr_html_()`), mounted as iframe.srcdoc so its
    # IDs/CSS don't collide with the editor. Empty string ⇒ no section.
    if tally_plot_html:
        tally_plot_srcdoc = (
            tally_plot_html.replace("&", "&amp;")
            .replace('"', "&quot;")
            .replace("<", "&lt;")
        )
    else:
        tally_plot_srcdoc = ""

    # The initial result-panel text is substituted as the body of a <pre>,
    # so HTML-escape the special characters.
    initial_result = initial_result_text if initial_result_text else "No result yet."
    initial_result_escaped = (
        initial_result.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )

    html = (
        _HTML_TEMPLATE.read_text(encoding="utf-8")
        .replace("__APP_JS__", _APP_JS.read_text(encoding="utf-8"))
        .replace("__WASM_B64__", wasm_b64)
        .replace("__JS_SRC_JSON__", json.dumps(js_glue_src))
        .replace("__MODEL_JSON__", json.dumps(model_json))
        .replace("__EMBEDDED_XS__", embedded_xs_json)
        .replace("__EMBEDDED_PHOTON_XS__", embedded_photon_json)
        .replace("__SECTION_FLAGS__", section_flags_json)
        .replace("__TALLY_PLOT_SRCDOC__", tally_plot_srcdoc)
        .replace("__INITIAL_RESULT_TEXT__", initial_result_escaped)
    )
    out.write_text(html, encoding="utf-8")
    return out
