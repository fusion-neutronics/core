#!/usr/bin/env python3
"""Generate and post-process the type stubs for both wheels.

1. Runs `cargo run -p <bindings crate> --bin stub_gen` to emit
   `<package root>/_core/__init__.pyi` from the PyO3 bindings' type metadata
   and Rust doc comments.
2. Relocates the functions that live in runtime submodules (`data`,
   `_decay_photons`, ...) into `_core/<submodule>.pyi`. pyo3-stub-gen cannot
   namespace a `#[pyfunction]` into a submodule (the submodule is built at
   runtime in lib.rs), so it emits them at the `_core` top level; this step
   moves them to match the runtime layout (`from yamc._core.data import ...`).

Two packages come out of this repo: `yamc` (transport plus everything) and
`yani` (transmutation only, issue #381). They share their bindings crate, so
they share this script; PACKAGES below is the only place they differ.

Idempotent: running it twice produces the same tree. CI runs this then
`git diff --exit-code` to detect drift.
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# name -> runtime submodule (see crates/yamc-python/src/lib.rs add_submodule).
# Entries may be functions OR classes; relocate_submodules moves either.
SUBMODULES: dict[str, list[str]] = {
    "data": [
        "dose_coefficients",
        "mass_attenuation_coefficient",
        "mass_energy_absorption_coefficient",
        "natural_abundance",
        "element_nuclides",
        "element_names",
        "reaction_names",
        "clear_nuclide_cache",
        "split_nuclide",
        "atomic_symbol",
        "atomic_number",
    ],
    # Lump shapes for self-shielding, constructed by the caller and handed in.
    "shapes": [
        "SphereLump",
        "CubeLump",
        "FoilLump",
        "CylinderLump",
        "WireLump",
    ],
    # Source-building distributions (classes) + the fusion-spectrum helper.
    "sources": [
        "Discrete",
        "Histogram",
        "Uniform",
        "Normal",
        "Isotropic",
        "Monodirectional",
        "CylindricalRing",
        "fusion_neutron_spectrum",
        "tokamak_source",
        "tokamak_ion_density",
        "tokamak_ion_temperature",
        "tokamak_convert_a_alpha_to_r_z",
        "tokamak_neutron_source_density",
    ],
    "parallel": [
        "has_mpi",
        "gpu_available",
        "list_gpu_adapters",
        "mpi_rank",
        "mpi_size",
        "mpi_finalize",
    ],
    "_decay_photons": [
        "get_radionuclides_from_chain",
    ],
    # Curated material collections. `pnnl` itself is a MaterialCollection
    # instance added in lib.rs, so it has no stub-gen metadata and is
    # declared in the submodule stub separately (see SUBMODULE_EXTRA).
    "materials": [
        "MaterialCollection",
        "NameIter",
        "collections",
    ],
}

# Extra source appended verbatim to a submodule's stub, for objects that are
# instantiated in lib.rs rather than declared with a #[pyclass]/#[pyfunction]
# and so carry no pyo3-stub-gen metadata.
SUBMODULE_EXTRA: dict[str, str] = {
    "materials": (
        "pnnl: MaterialCollection\n"
        'r"""The PNNL Compendium of Material Composition Data for Radiation\n'
        "Transport Modeling (PNNL-15870, Rev. 2): 410 materials keyed by the\n"
        'report\'s own names."""\n'
    ),
}

def _m(sig: str, doc: str) -> str:
    return f'    def {sig}:\n        r"""{doc}"""'


_VTKHDF = ("to_vtkhdf", _m(
    "to_vtkhdf(self, filename: builtins.str, **kwargs: typing.Any) -> None",
    "Write to a VTK-HDF file for ParaView. Requires h5py (pip install yamc[viz])."))

# Methods added to _core classes at runtime in packages/yamc-core/python/yamc/__init__.py (monkey-
# patched, so invisible to pyo3-stub-gen) plus the .plot()/interactive_plot()
# methods skipped in Rust (pyo3-stub-gen Option<&str>-default bug). Injected
# here with import-clean signatures. class -> [(method_name, source_block)].
INJECTED_METHODS: dict[str, list[tuple[str, str]]] = {
    "Geometry": [_VTKHDF, ("plot", _m(
        "plot(self, *args: typing.Any, **kwargs: typing.Any) -> InteractivePlot",
        "Render an interactive geometry slice plot. See the docs for keyword arguments."))],
    "Cell": [_VTKHDF, ("plot", _m(
        "plot(self, *args: typing.Any, **kwargs: typing.Any) -> InteractivePlot",
        "Render an interactive plot of this cell. See the docs for keyword arguments."))],
    "Region": [_VTKHDF, ("plot", _m(
        "plot(self, *args: typing.Any, **kwargs: typing.Any) -> InteractivePlot",
        "Render an interactive plot of this region. See the docs for keyword arguments."))],
    "Halfspace": [_VTKHDF],
    "MeshGeometry": [("plot", _m(
        "plot(self, *args: typing.Any, **kwargs: typing.Any) -> InteractivePlot",
        "Render an interactive plot of this mesh geometry. See the docs for keyword arguments."))],
    "NeutronSource": [_VTKHDF],
    "PhotonSource": [_VTKHDF],
    "Tally": [_VTKHDF, ("interactive_plot", _m(
        "interactive_plot(self, *args: typing.Any, **kwargs: typing.Any) -> InteractiveTallyPlot",
        "Render an interactive mesh-tally plot. See the docs for keyword arguments."))],
    "Model": [
        _VTKHDF,
        ("to_html", _m(
            "to_html(self, path: builtins.str, embed_cross_sections: builtins.bool | typing.Iterable[builtins.str] = False, **kwargs: typing.Any) -> typing.Any",
            "Export a self-contained, browser-runnable interactive HTML page. Returns the written path.")),
        ("plot", _m(
            "plot(self, *args: typing.Any, **kwargs: typing.Any) -> InteractivePlot",
            "Render an interactive geometry slice plot. See the docs for keyword arguments.")),
        # `radionuclides` is no longer injected: it is a real #[pymethod] on
        # PyModel now, so pyo3-stub-gen emits it with its own docstring.
    ],
    "TallyResult": [
        ("to_xarray", _m(
            'to_xarray(self, field: typing.Literal["mean", "standard_deviation", "relative_error", "variance", "total_count"] = "mean") -> typing.Any',
            "Return one tally field as a labelled xarray.DataArray. Requires xarray (pip install yamc[xarray]).")),
        ("to_dataset", _m(
            "to_dataset(self) -> typing.Any",
            "Return mean/std_dev/relative_error as one labelled xarray.Dataset. Requires xarray.")),
    ],
}

SUBMOD_HEADER = (
    "# Generated by scripts/build_stubs.py -- do not edit.\n"
    "# Relocated from _core/__init__.pyi to match the runtime submodule layout.\n"
    "# ruff: noqa: E501, F401, F403, F405\n"
    "import builtins\n"
    "import typing\n"
    # Relocated submodule functions may reference _core classes in their
    # signatures (e.g. data.dose_coefficients -> DoseCoefficients,
    # sources.tokamak_source -> NeutronSource); import them so the moved
    # annotations still resolve. Unused names in other submodule stubs are
    # covered by the F401 noqa above.
    "from {package}._core import (\n"
    "    DoseCoefficients,\n"
    "    Material,\n"
    "    NeutronSource,\n"
    "    PhotonCoefficients,\n"
    ")\n"
)


# Members the bindings crate defines but a given wheel does not register.
# pyo3-stub-gen collects every `#[gen_stub_pyclass]` in the crate through
# `inventory`, with no idea which `add_class` calls a particular `#[pymodule]`
# actually made, so a stub for the transmutation wheel would otherwise advertise
# the transport-only surface that `register_classes` deliberately skips for it
# (issue #452). check_public_surface.py fails on exactly this drift.
OMITTED: dict[str, set[str]] = {
    "yani": {
        "AngleDistribution",
        "Particle",
        "PhotonSource",
        "ReactionProduct",
        "Tabulated",
        "create_test_reaction_product",
        "sample_scatter_cosine",
    },
}


@dataclass
class Package:
    """One wheel's stub tree: where it lives and what its runtime shape is."""

    name: str
    core: Path
    crate: str
    features: str
    submodules: dict[str, list[str]]
    submodule_extra: dict[str, str]
    injected: dict[str, list[tuple[str, str]]]

    @property
    def init(self) -> Path:
        return self.core / "__init__.pyi"


def regenerate(pkg: Package) -> None:
    subprocess.run(
        ["cargo", "run", "--quiet", "-p", pkg.crate, "--bin", "stub_gen",
         "--features", pkg.features],
        cwd=ROOT, check=True,
    )


def relocate_submodules(pkg: Package) -> None:
    SUBMODULES, SUBMODULE_EXTRA, CORE, INIT = (
        pkg.submodules, pkg.submodule_extra, pkg.core, pkg.init)
    src = INIT.read_text()
    tree = ast.parse(src)
    fn_to_mod = {fn: mod for mod, fns in SUBMODULES.items() for fn in fns}

    segments: dict[str, str] = {}
    drop_lines: set[int] = set()  # 1-indexed lines to remove from __init__.pyi
    src_lines = src.splitlines()
    for node in tree.body:
        # Relocate both functions and classes (e.g. the yamc.sources
        # distributions). ast.get_source_segment omits decorators, so slice the
        # source from the first decorator line to include e.g. @typing.final.
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in fn_to_mod:
            start = node.lineno
            if node.decorator_list:
                start = min(start, min(d.lineno for d in node.decorator_list))
            segments[node.name] = "\n".join(src_lines[start - 1 : node.end_lineno])
            drop_lines.update(range(start, node.end_lineno + 1))

    missing = set(fn_to_mod) - set(segments)
    if missing:
        sys.exit(f"build_stubs: expected submodule fns not found in stub: {sorted(missing)}")

    # Write each submodule stub.
    for mod, fns in SUBMODULES.items():
        body = "\n\n".join(segments[fn] for fn in fns)
        extra = SUBMODULE_EXTRA.get(mod, "")
        names = fns + ([extra.split(":", 1)[0]] if extra else [])
        all_list = "".join(f'    "{name}",\n' for name in names)
        header = SUBMOD_HEADER.format(package=pkg.name)
        text = f"{header}\n__all__ = [\n{all_list}]\n\n{body}\n"
        if extra:
            text += f"\n{extra}"
        (CORE / f"{mod}.pyi").write_text(text)

    # Strip the relocated funcs from __init__.pyi and from its __all__.
    relocated = set(fn_to_mod)
    out = []
    for i, line in enumerate(src.splitlines(keepends=True), start=1):
        if i in drop_lines:
            continue
        m = re.match(r'\s*"([^"]+)",\s*$', line)
        if m and m.group(1) in relocated:
            continue
        out.append(line)
    new = "".join(out)

    # Expose the submodules so `from <pkg>._core import data` resolves.
    imports = "".join(f"from . import {mod} as {mod}\n" for mod in SUBMODULES)
    new = new.replace("import builtins\nimport typing\n",
                      f"import builtins\nimport typing\n{imports}", 1)

    new = re.sub(r"\n{3,}", "\n\n", new)
    INIT.write_text(new)


def omit_members(pkg: Package) -> None:
    """Un-advertise the members this wheel does not register.

    Only the ``__all__`` entries go. The class declarations stay, because
    signatures that DO belong to this wheel refer to them -- ``Reaction.products``
    returns ``list[ReactionProduct]`` and ``NeutronSource.sample`` returns
    ``Particle`` in both wheels. Those objects are real here; what is not real is
    the top-level name, since ``register_classes`` never binds it (#452). A
    pyclass is usable without ``add_class``; ``add_class`` only puts the name in
    the module namespace.
    """
    omit = OMITTED.get(pkg.name)
    if not omit:
        return
    src = pkg.init.read_text()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    found = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in omit
    }
    missing = omit - found
    if missing:
        sys.exit(
            f"build_stubs: {pkg.name} omits {sorted(missing)}, but the stub does "
            "not define them -- the list is stale, or the name changed"
        )
    out = []
    for line in lines:
        m = re.match(r'\s*"([^"]+)",\s*$', line)
        if m and m.group(1) in omit:
            continue
        out.append(line)
    pkg.init.write_text("".join(out))


def inject_methods(pkg: Package) -> None:
    INJECTED_METHODS, INIT = pkg.injected, pkg.init
    src = INIT.read_text()
    tree = ast.parse(src)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    lines = src.splitlines()

    inserts: list[tuple[int, str]] = []  # (insert-after 1-indexed line, block)
    for cls_name, methods in INJECTED_METHODS.items():
        cls = classes.get(cls_name)
        if cls is None:
            sys.exit(f"build_stubs: class {cls_name} not found for method injection")
        existing = {n.name for n in cls.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        blocks = [blk for (name, blk) in methods if name not in existing]
        if blocks:
            inserts.append((cls.end_lineno, "\n".join(blocks)))

    # Apply bottom-up so earlier line numbers stay valid.
    for end_lineno, block in sorted(inserts, key=lambda x: -x[0]):
        lines.insert(end_lineno, block)
    INIT.write_text("\n".join(lines) + "\n")


# The bindings crate is shared, and its doc comments are written from the
# transport package's side: every doctest says `import yamc`, and every worked
# example reaches for `yamc.Material`. Emitted verbatim into the other wheel's
# stubs, those become the examples on ITS published API reference -- yani's
# reference told readers to import a package they had not installed, in 64
# places, because mkdocstrings renders these docstrings straight onto the page.
#
# So the package name is rewritten to match the wheel the stub belongs to. Only
# where the name behind it is actually part of that wheel: `yamc.Tally` and
# `yamc.Model` are transport objects yani does not have, and a yani docstring
# pointing at them is a genuine cross-package reference that should keep saying
# yamc. Which way each reference went is printed, so a new one cannot slip in
# unnoticed.
CANONICAL = "yamc"  # the name the shared doc comments are written against


def package_surface(pkg: Package) -> set[str]:
    """Every name this wheel binds, from the stubs just written for it.

    Read back rather than hardcoded: this runs after relocation and omission,
    so the stub tree is already the authority on what the wheel exposes.
    """
    names: set[str] = set()
    for path in sorted(pkg.core.glob("*.pyi")):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(getattr(t, "id", None) == "__all__" for t in node.targets):
                continue
            if isinstance(node.value, ast.List):
                names.update(el.value for el in node.value.elts
                             if isinstance(el, ast.Constant))
        # Submodule members are referenced bare in the prose (`yamc.Discrete`,
        # not `yamc.sources.Discrete`), so a flat set is what matches.
        names.add(path.stem)
    names.discard("__init__")

    # The five nuclear-data settings are module-level annotated globals in the
    # hand-maintained package stub, not `__all__` entries in `_core`, and the
    # doc comments reach for them by name (`yamc.cross_section_data`).
    package_stub = pkg.core.parent / "__init__.pyi"
    if package_stub.exists():
        for node in ast.parse(package_stub.read_text()).body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def rewrite_package_references(pkg: Package) -> None:
    if pkg.name == CANONICAL:
        return
    surface = package_surface(pkg)
    # `yamc.foo` keeps its package name when `foo` is not ours; bare `yamc`
    # ("import yamc", "yamc's own loader") is always this wheel talking about
    # itself.
    pattern = re.compile(rf"\b{CANONICAL}\b(\.([A-Za-z_][A-Za-z0-9_]*))?")
    foreign: dict[str, int] = {}

    def swap(m: re.Match[str]) -> str:
        attr = m.group(2)
        if attr is not None and attr not in surface:
            foreign[f"{CANONICAL}.{attr}"] = foreign.get(f"{CANONICAL}.{attr}", 0) + 1
            return m.group(0)
        return pkg.name + (m.group(1) or "")

    changed = 0
    for path in sorted(pkg.core.glob("*.pyi")):
        src = path.read_text()
        out = pattern.sub(swap, src)
        if out != src:
            changed += len(pattern.findall(src)) - sum(foreign.values())
            path.write_text(out)

    if foreign:
        kept = ", ".join(f"{name} x{n}" for name, n in sorted(foreign.items()))
        print(f"  {pkg.name}: kept {sum(foreign.values())} cross-package "
              f"reference(s): {kept}")


PACKAGES = [
    Package(
        name="yamc",
        core=ROOT / "packages" / "yamc-core" / "python" / "yamc" / "_core",
        crate="yamc-python",
        features="stub-gen,mesh,cad",
        submodules=SUBMODULES,
        submodule_extra=SUBMODULE_EXTRA,
        injected=INJECTED_METHODS,
    ),
    # yani has no transport, so no `parallel` submodule and none of the
    # monkey-patched plotting/vtkhdf methods to declare.
    Package(
        name="yani",
        core=ROOT / "packages" / "yani-core" / "python" / "yani" / "_core",
        crate="yani-python",
        features="stub-gen",
        # No transport, so no `parallel`.
        submodules={k: v for k, v in SUBMODULES.items() if k != "parallel"},
        submodule_extra=SUBMODULE_EXTRA,
        injected={},
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-regen", action="store_true",
                    help="skip cargo stub_gen; only post-process existing stub")
    ap.add_argument("--package", choices=[p.name for p in PACKAGES],
                    help="only this package (default: all)")
    args = ap.parse_args()
    for pkg in PACKAGES:
        if args.package and pkg.name != args.package:
            continue
        if not args.no_regen:
            regenerate(pkg)
        relocate_submodules(pkg)
        omit_members(pkg)
        inject_methods(pkg)
        rewrite_package_references(pkg)
        print(f"stubs written under {pkg.core.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
