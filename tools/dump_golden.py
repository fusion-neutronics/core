# SPDX-License-Identifier: MIT
"""Dump golden files from the Python reader for the Rust port to check against.

The Rust crate is being ported file by file, and the only thing that makes that
safe is holding it to what the Python reader already produces. This writes that
reference out in a format the Rust test reads without pulling in a JSON
dependency.

Every value is emitted as one line::

    V <path> <F|I|T> <values...>

where the path names the value (``0/3/1/sigma/x`` is the MF=3 MT=1 cross
section abscissae of the first material) and the tag says how to read the rest:
``F`` floats, ``I`` integers, ``T`` a hex-encoded string. Strings are hex
because ENDF text fields are fixed-width and carry significant spaces, which a
whitespace-separated format would eat.

The Rust test builds the same map from its own parse and compares the two whole.
A path present on one side and not the other fails, so a field that is renamed,
dropped or added is caught as loudly as one whose value is wrong.

Floats are written with ``repr``, the shortest string that round-trips to the
same double. Rust's float parser is correctly rounded, so it recovers
bit-identical values and the comparison is exact rather than approximate.

Regenerate every golden file::

    python tools/dump_golden.py

Add a fixture and regenerate only it::

    python tools/dump_golden.py tests/data/n-092_U_235.endf

Check that the stored goldens still match what the reader produces, without
writing anything::

    python tools/dump_golden.py --check

The check compares the *text*, not the compressed bytes: two xz encoders can
write the same content differently, so a byte comparison fails on a machine
whose liblzma differs from the one that wrote the file. A real difference is
reported as a diff of the offending lines.
"""

from __future__ import annotations

import difflib
import io
import lzma
import sys
from pathlib import Path

import endf

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "crates" / "endf" / "tests" / "golden"

#: Where ENDF fixtures are looked for when no argument is given.
#:
#: Upstream keeps these in `tests/` at the repository root. Here that name is a
#: symlink to `crates/yamc/tests`, so they live inside the crate instead. The
#: SOURCE line each golden records is written relative to ROOT, and
#: `golden.rs::repo_root()` resolves it the same way, so moving the fixtures
#: needs this line changed and nothing on the Rust side.
FIXTURE_DIRS = [ROOT / "crates" / "endf" / "fixtures"]

#: Fixtures are stored xz-compressed (an evaluation is highly repetitive and
#: compresses about six to one) so the suffix to match on is the one before
#: the `.xz`.
FIXTURE_SUFFIXES = {".endf", ".dat", ".ace"}


def fixture_kind(path: Path) -> str:
    """The format suffix of a fixture, ignoring any `.xz`."""
    if path.suffix == ".xz":
        path = path.with_suffix("")
    return path.suffix.lower()


#: How many interior sample points to record per tabulated function, on top of
#: the ones the region boundaries force. Enough to exercise each interpolation
#: law without the golden file growing to the size of the evaluation.
MAX_SAMPLES = 24


class Dump:
    """Collects `path -> value` lines."""

    def __init__(self, out):
        self.out = out

    def floats(self, path: str, values) -> None:
        self.out.write(f"V {path} F " + " ".join(repr(float(v)) for v in values) + "\n")

    def float(self, path: str, value) -> None:
        self.floats(path, [value])

    def ints(self, path: str, values) -> None:
        self.out.write(f"V {path} I " + " ".join(str(int(v)) for v in values) + "\n")

    def int(self, path: str, value) -> None:
        self.ints(path, [value])

    def text(self, path: str, value: str) -> None:
        self.out.write(f"V {path} T {value.encode().hex()}\n")

    def tab1(self, path: str, table) -> None:
        """A TAB1: its tabulation, and the interpolation sampled."""
        self.floats(f"{path}/x", table.x)
        self.floats(f"{path}/y", table.y)
        self.ints(f"{path}/bp", table.breakpoints)
        self.ints(f"{path}/int", table.interpolation)
        points = sample_points(table)
        if points:
            self.floats(f"{path}/evalx", points)
            self.floats(f"{path}/evaly", [table(p) for p in points])

    def tab2(self, path: str, table) -> None:
        self.ints(f"{path}/bp", table.breakpoints)
        self.ints(f"{path}/int", table.interpolation)


def sample_points(table) -> list[float]:
    """Points at which to pin the interpolation.

    Bin midpoints, because a tabulated point returns its own value whatever the
    interpolation law is and so proves nothing. The bins on either side of a
    region boundary are always included: picking the scheme for the bin a point
    falls in is the part of the evaluation most likely to be got wrong.
    """
    x = table.x
    if len(x) < 2:
        return [float(v) for v in x]

    def midpoint(i: int) -> float:
        return float(0.5 * (x[i] + x[i + 1]))

    wanted = set()
    for b in table.breakpoints:
        for i in (int(b) - 2, int(b) - 1):
            if 0 <= i < len(x) - 1:
                wanted.add(i)
    n_bins = len(x) - 1
    step = max(1, n_bins // MAX_SAMPLES)
    wanted.update(range(0, n_bins, step))

    points = [midpoint(i) for i in sorted(wanted)]
    # Both ends, and outside them, to pin the clamping behaviour.
    return [float(x[0]) * 0.5, float(x[0]), *points, float(x[-1]), float(x[-1]) * 2.0]


# --------------------------------------------------------------------------
# One dumper per ENDF file the Rust crate can parse. Extend as the port goes.
# --------------------------------------------------------------------------


def dump_nu(d: Dump, path: str, section: dict) -> None:
    """A nu-bar, in whichever of the two forms the evaluation uses."""
    if "C" in section:
        d.floats(f"{path}/poly", section["C"])
    if "nu" in section:
        d.tab1(f"{path}/tab", section["nu"])


def dump_mf1(d: Dump, path: str, mt: int, section: dict) -> None:
    if mt == 451:
        for key in (
            "ZA",
            "LRP",
            "LFI",
            "NLIB",
            "NMOD",
            "LIS",
            "LISO",
            "NFOR",
            "LREL",
            "NSUB",
            "NVER",
            "LDRV",
            "NWD",
            "NXC",
        ):
            d.int(f"{path}/{key}", section[key])
        for key in ("AWR", "ELIS", "STA", "AWI", "EMAX", "TEMP"):
            d.float(f"{path}/{key}", section[key])
        if section.get("ZSYMAM") is not None:
            for key in (
                "ZSYMAM",
                "ALAB",
                "EDATE",
                "AUTH",
                "REF",
                "DDATE",
                "RDATE",
                "ENDATE",
            ):
                d.text(f"{path}/{key}", section[key])
            for i, line in enumerate(section["HSUB"]):
                d.text(f"{path}/HSUB/{i}", line)
            for i, line in enumerate(section["description"]):
                d.text(f"{path}/description/{i}", line)
        for i, entry in enumerate(section["section_list"]):
            d.ints(f"{path}/section_list/{i}", entry)

    elif mt in (452, 456):
        d.int(f"{path}/ZA", section["ZA"])
        d.float(f"{path}/AWR", section["AWR"])
        d.int(f"{path}/LNU", section["LNU"])
        dump_nu(d, path, section)

    elif mt == 455:
        d.int(f"{path}/ZA", section["ZA"])
        d.float(f"{path}/AWR", section["AWR"])
        d.int(f"{path}/LDG", section["LDG"])
        d.int(f"{path}/LNU", section["LNU"])
        if "lambda" in section:
            d.floats(f"{path}/lambda", section["lambda"])
        if "E_int" in section:
            d.tab2(f"{path}/E_int", section["E_int"])
        for i, c in enumerate(section.get("constants", [])):
            d.float(f"{path}/constants/{i}/E", c["E"])
            d.floats(f"{path}/constants/{i}/lambda", c["lambda"])
            d.floats(f"{path}/constants/{i}/alpha", c["alpha"])
        dump_nu(d, path, section)

    elif mt == 458:
        # ZA comes from a CONT rather than a HEAD record here, so it is a float
        # in this section and an int everywhere else. See issue #14.
        d.float(f"{path}/ZA", section["ZA"])
        d.float(f"{path}/AWR", section["AWR"])
        d.int(f"{path}/LFC", section["LFC"])
        d.int(f"{path}/NPLY", section["NPLY"])
        if "NFC" in section:
            d.int(f"{path}/NFC", section["NFC"])
        for name in ("EFR", "ENP", "END", "EGP", "EGD", "EB", "ENU", "ER", "ET"):
            value = section[name]
            if isinstance(value, dict):
                d.int(f"{path}/{name}/LDRV", value["LDRV"])
                d.tab1(f"{path}/{name}/EIFC", value["EIFC"])
            else:
                # list of (coefficient, uncertainty) pairs
                d.floats(f"{path}/{name}/poly", [v for pair in value for v in pair])

    elif mt == 460:
        d.int(f"{path}/ZA", section["ZA"])
        d.float(f"{path}/AWR", section["AWR"])
        d.int(f"{path}/LO", section["LO"])
        if "NG" in section:
            d.int(f"{path}/NG", section["NG"])
        if "E" in section:
            d.floats(f"{path}/E", section["E"])
        for i, t in enumerate(section.get("T", [])):
            d.tab1(f"{path}/T/{i}", t)
        if "lambda" in section:
            d.floats(f"{path}/lambda", section["lambda"])


def dump_mf2(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.int(f"{path}/NIS", section["NIS"])
    for a, iso in enumerate(section["isotopes"]):
        ip = f"{path}/isotopes/{a}"
        d.float(f"{ip}/ZAI", iso["ZAI"])
        d.float(f"{ip}/ABN", iso["ABN"])
        d.int(f"{ip}/LFW", iso["LFW"])
        d.int(f"{ip}/NER", iso["NER"])
        for b, r in enumerate(iso["ranges"]):
            rp = f"{ip}/ranges/{b}"
            d.float(f"{rp}/EL", r["EL"])
            d.float(f"{rp}/EH", r["EH"])
            for key in ("LRU", "LRF", "NRO", "NAPS"):
                d.int(f"{rp}/{key}", r[key])
            dump_mf2_parameters(d, rp, r)


def dump_mf2_parameters(d: Dump, rp: str, r: dict) -> None:
    """Whatever representation the range turned out to use.

    Driven by which keys are present rather than by LRU/LRF, so that a range
    the dispatch skips (issue #15) emits nothing and the Rust side, which
    reproduces the same skip, emits nothing either.
    """
    if "APE" in r:
        d.tab1(f"{rp}/APE", r["APE"])
    for key in ("SPI", "AP"):
        if key in r:
            d.float(f"{rp}/{key}", r[key])
    for key in ("NLS", "LAD", "NLSC", "LSSF", "IFG", "KRM", "NJS", "KRL", "NPP", "NE"):
        if key in r:
            d.int(f"{rp}/{key}", r[key])
    if "ES" in r:
        d.floats(f"{rp}/ES", r["ES"])

    # Resolved: Breit-Wigner and Reich-Moore both key off 'sections'.
    for i, s in enumerate(r.get("sections", [])):
        sp = f"{rp}/sections/{i}"
        for key in ("AWRI", "QX", "APL"):
            if key in s:
                d.float(f"{sp}/{key}", s[key])
        for key in ("L", "LRX", "NRS"):
            if key in s:
                d.int(f"{sp}/{key}", s[key])
        for key in ("ER", "AJ", "GT", "GN", "GG", "GF", "GFA", "GFB"):
            if key in s:
                d.floats(f"{sp}/{key}", s[key])

    # R-matrix limited.
    if "particle_pairs" in r:
        for key, values in r["particle_pairs"].items():
            d.floats(f"{rp}/particle_pairs/{key}", values)
    for i, g in enumerate(r.get("spin_groups", [])):
        gp = f"{rp}/spin_groups/{i}"
        for key in ("AJ", "PJ"):
            d.float(f"{gp}/{key}", g[key])
        for key in ("KBK", "KPS", "NCH", "NRS", "NX", "LCH", "LBK", "LPS"):
            if key in g:
                d.int(f"{gp}/{key}", g[key])
        for key, values in g["channels"].items():
            d.floats(f"{gp}/channels/{key}", values)
        d.floats(f"{gp}/ER", g["ER"])
        for c, row in enumerate(g["GAM"]):
            d.floats(f"{gp}/GAM/{c}", row)
        for key in ("ED", "EU"):
            if key in g:
                d.float(f"{gp}/{key}", g[key])
        for key in ("RBR", "RBI", "PSR", "PSI"):
            if key in g:
                d.tab1(f"{gp}/{key}", g[key])

    # Unresolved.
    for i, u in enumerate(r.get("ranges", [])):
        up = f"{rp}/ranges/{i}"
        d.float(f"{up}/AWRI", u["AWRI"])
        d.int(f"{up}/L", u["L"])
        d.int(f"{up}/NJS", u["NJS"])
        for key in ("D", "AJ", "AMUN", "GNO", "GG"):
            if key in u:
                d.floats(f"{up}/{key}", u[key])
        for j, p in enumerate(u.get("parameters", [])):
            pp = f"{up}/parameters/{j}"
            for key in ("MUF", "INT", "NE"):
                if key in p:
                    d.int(f"{pp}/{key}", p[key])
            for key in ("D", "AJ", "AMUN", "AMUX", "AMUF", "GN0", "GG"):
                if key in p:
                    value = p[key]
                    if hasattr(value, "__len__"):
                        d.floats(f"{pp}/{key}", value)
                    else:
                        d.float(f"{pp}/{key}", value)
            for key in ("E", "GX", "GF"):
                if key in p:
                    d.floats(f"{pp}/{key}", p[key])


def dump_mf3(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.float(f"{path}/QM", section["QM"])
    d.float(f"{path}/QI", section["QI"])
    d.int(f"{path}/LR", section["LR"])
    d.tab1(f"{path}/sigma", section["sigma"])


def dump_mf4(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    for key in ("LTT", "LI", "LCT"):
        d.int(f"{path}/{key}", section[key])
    for kind in ("legendre", "tabulated"):
        if kind not in section:
            continue
        sub = section[kind]
        sp = f"{path}/{kind}"
        d.tab2(f"{sp}/E_int", sub["E_int"])
        d.float(f"{sp}/T", sub["T"])
        d.int(f"{sp}/LT", sub["LT"])
        d.floats(f"{sp}/E", sub["E"])
        for i, a in enumerate(sub.get("a_l", [])):
            d.floats(f"{sp}/a_l/{i}", a)
        for i, mu in enumerate(sub.get("mu", [])):
            d.tab1(f"{sp}/mu/{i}", mu)

    # The interpreted form, which is what a consumer actually uses.
    from endf.mf4 import AngleDistribution

    dist = AngleDistribution.from_dict(section)
    dump_angle_distribution(d, f"{path}/angle", dist)
    if len(dist.energy) > 0:
        for cutoff in (-1.0, -0.5, 0.0, 0.5):
            name = f"{cutoff:+.1f}"
            d.floats(
                f"{path}/angle/forward_fraction/{name}", dist.forward_fraction(cutoff)
            )


def dump_angle_distribution(d: Dump, path: str, dist) -> None:
    """An :class:`endf.mf4.AngleDistribution`, whichever shapes it holds."""
    from numpy.polynomial import Legendre

    from endf.function import Tabulated1D

    d.floats(f"{path}/energy", dist.energy)
    d.int(f"{path}/n_mu", len(dist.mu))
    for i, mu in enumerate(dist.mu):
        p = f"{path}/mu/{i}"
        if isinstance(mu, Legendre):
            d.text(f"{p}/kind", "legendre")
            d.floats(f"{p}/coef", mu.coef)
        elif isinstance(mu, Tabulated1D):
            d.text(f"{p}/kind", "tabulated")
            d.tab1(f"{p}/f", mu)
        else:
            dump_univariate(d, p, mu)


def dump_univariate(d: Dump, p: str, u) -> None:
    """One :class:`endf.univariate.Univariate`, whichever shape it is."""
    from endf.univariate import Discrete, Mixture, Tabular, Uniform

    if isinstance(u, Discrete):
        d.text(f"{p}/kind", "discrete")
        d.floats(f"{p}/x", u.x)
        d.floats(f"{p}/p", u.p)
        d.floats(f"{p}/cdf", u.cdf())
    elif isinstance(u, Tabular):
        d.text(f"{p}/kind", "tabular")
        d.text(f"{p}/interpolation", u.interpolation)
        d.floats(f"{p}/x", u.x)
        d.floats(f"{p}/p", u.p)
        d.floats(f"{p}/cdf", u.cdf())
    elif isinstance(u, Uniform):
        d.text(f"{p}/kind", "uniform")
        d.float(f"{p}/a", u.a)
        d.float(f"{p}/b", u.b)
    elif isinstance(u, Mixture):
        d.text(f"{p}/kind", "mixture")
        d.floats(f"{p}/probability", u.probability)
        for j, sub in enumerate(u.distribution):
            dump_univariate(d, f"{p}/distribution/{j}", sub)
    else:
        raise TypeError(f"unexpected distribution {type(u)}")

    # The CDF as the file gave it, where there was one.
    if getattr(u, "c", None) is not None:
        d.floats(f"{p}/c", u.c)


def dump_energy_distribution(d: Dump, p: str, dist) -> None:
    """One energy distribution object, of whichever law."""
    from endf.mf5 import (
        ArbitraryTabulated,
        ContinuousTabular,
        DiscretePhoton,
        Evaporation,
        GeneralEvaporation,
        LevelInelastic,
        MadlandNix,
        MaxwellEnergy,
        WattEnergy,
    )

    if isinstance(dist, ArbitraryTabulated):
        d.text(f"{p}/kind", "arbitrary-tabulated")
        d.floats(f"{p}/E", dist.energy)
        for j, g in enumerate(dist.pdf):
            d.tab1(f"{p}/g/{j}", g)
    elif isinstance(dist, GeneralEvaporation):
        d.text(f"{p}/kind", "general-evaporation")
        d.float(f"{p}/U", dist.u)
        d.tab1(f"{p}/theta", dist.theta)
        d.tab1(f"{p}/g", dist.g)
    elif isinstance(dist, MaxwellEnergy):
        d.text(f"{p}/kind", "maxwell")
        d.float(f"{p}/U", dist.u)
        d.tab1(f"{p}/theta", dist.theta)
    elif isinstance(dist, Evaporation):
        d.text(f"{p}/kind", "evaporation")
        d.float(f"{p}/U", dist.u)
        d.tab1(f"{p}/theta", dist.theta)
    elif isinstance(dist, WattEnergy):
        d.text(f"{p}/kind", "watt")
        d.float(f"{p}/U", dist.u)
        d.tab1(f"{p}/a", dist.a)
        d.tab1(f"{p}/b", dist.b)
    elif isinstance(dist, MadlandNix):
        d.text(f"{p}/kind", "madland-nix")
        d.float(f"{p}/EFL", dist.efl)
        d.float(f"{p}/EFH", dist.efh)
        # The attribute is `tm`, though the format calls the field T_M.
        d.tab1(f"{p}/T_M", dist.tm)
    elif isinstance(dist, LevelInelastic):
        d.text(f"{p}/kind", "level-inelastic")
        d.float(f"{p}/threshold", dist.threshold)
        d.float(f"{p}/mass_ratio", dist.mass_ratio)
    elif isinstance(dist, DiscretePhoton):
        d.text(f"{p}/kind", "discrete-photon")
        d.int(f"{p}/primary_flag", dist.primary_flag)
        d.float(f"{p}/energy", dist.energy)
        d.float(f"{p}/atomic_weight_ratio", dist.atomic_weight_ratio)
    elif isinstance(dist, ContinuousTabular):
        d.text(f"{p}/kind", "continuous-tabular")
        d.ints(f"{p}/bp", dist.breakpoints)
        d.ints(f"{p}/int", dist.interpolation)
        d.floats(f"{p}/E", dist.energy)
        for j, eout in enumerate(dist.energy_out):
            dump_univariate(d, f"{p}/energy_out/{j}", eout)
    else:
        raise TypeError(f"unexpected energy distribution {type(dist)}")


def dump_angle_energy(d: Dump, p: str, ae) -> None:
    """One joint angle-energy distribution, of whichever shape."""
    from endf.angle_energy import (
        CorrelatedAngleEnergy,
        KalbachMann,
        NBodyPhaseSpace,
        UncorrelatedAngleEnergy,
    )

    if isinstance(ae, UncorrelatedAngleEnergy):
        d.text(f"{p}/kind", "uncorrelated")
        if ae.angle is not None:
            dump_angle_distribution(d, f"{p}/angle", ae.angle)
        if ae.energy is not None:
            dump_energy_distribution(d, f"{p}/energy", ae.energy)
    elif isinstance(ae, KalbachMann):
        d.text(f"{p}/kind", "kalbach-mann")
        d.ints(f"{p}/bp", ae.breakpoints)
        d.ints(f"{p}/int", ae.interpolation)
        d.floats(f"{p}/E", ae.energy)
        for j, eout in enumerate(ae.energy_out):
            dump_univariate(d, f"{p}/energy_out/{j}", eout)
        for j, r in enumerate(ae.precompound):
            d.tab1(f"{p}/precompound/{j}", r)
        for j, a in enumerate(ae.slope):
            d.tab1(f"{p}/slope/{j}", a)
    elif isinstance(ae, CorrelatedAngleEnergy):
        d.text(f"{p}/kind", "correlated")
        d.ints(f"{p}/bp", ae.breakpoints)
        d.ints(f"{p}/int", ae.interpolation)
        d.floats(f"{p}/E", ae.energy)
        for j, eout in enumerate(ae.energy_out):
            dump_univariate(d, f"{p}/energy_out/{j}", eout)
        for j, mu_j in enumerate(ae.mu):
            for k, mu_jk in enumerate(mu_j):
                dump_univariate(d, f"{p}/mu/{j}/{k}", mu_jk)
    elif isinstance(ae, NBodyPhaseSpace):
        d.text(f"{p}/kind", "nbody")
        d.float(f"{p}/total_mass", ae.total_mass)
        d.int(f"{p}/n_particles", ae.n_particles)
        d.float(f"{p}/atomic_weight_ratio", ae.atomic_weight_ratio)
        d.float(f"{p}/q_value", ae.q_value)
    else:
        raise TypeError(f"unexpected angle-energy distribution {type(ae)}")


def dump_mf5(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.int(f"{path}/NK", section["NK"])
    for i, sub in enumerate(section["subsections"]):
        sp = f"{path}/subsections/{i}"
        d.int(f"{sp}/LF", sub["LF"])
        d.tab1(f"{sp}/p", sub["p"])
        dist = sub["distribution"]
        dp = f"{sp}/distribution"
        for key in ("U", "EFL", "EFH"):
            if key in dist:
                d.float(f"{dp}/{key}", dist[key])
        if "E_int" in dist:
            d.tab2(f"{dp}/E_int", dist["E_int"])
        if "E" in dist:
            d.floats(f"{dp}/E", dist["E"])
        # LF=1 stores a list under 'g'; LF=5 stores a single table there.
        if isinstance(dist.get("g"), list):
            for j, g in enumerate(dist["g"]):
                d.tab1(f"{dp}/g/{j}", g)
        elif "g" in dist:
            d.tab1(f"{dp}/g", dist["g"])
        for key in ("theta", "a", "b", "T_M"):
            if key in dist:
                d.tab1(f"{dp}/{key}", dist[key])


def dump_mf6(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    for key in ("JP", "LCT", "NK"):
        d.int(f"{path}/{key}", section[key])
    for i, p in enumerate(section["products"]):
        pp = f"{path}/products/{i}"
        d.int(f"{pp}/ZAP", p["ZAP"])
        d.float(f"{pp}/AWP", p["AWP"])
        d.int(f"{pp}/LIP", p["LIP"])
        d.int(f"{pp}/LAW", p["LAW"])
        d.tab1(f"{pp}/y_i", p["y_i"])
        if "distribution" not in p:
            continue
        dump_mf6_distribution(d, f"{pp}/distribution", p["distribution"])


def dump_mf6_distribution(d: Dump, dp: str, dist: dict) -> None:
    """One MF=6 distribution. Shared with MF=26, which reuses LAW=1 and LAW=2."""
    for key in ("LANG", "LEP", "NR", "NE", "LIDP", "NPSX"):
        if key in dist:
            d.int(f"{dp}/{key}", dist[key])
    for key in ("SPI", "APSX"):
        if key in dist:
            d.float(f"{dp}/{key}", dist[key])
    if "E_int" in dist:
        d.tab2(f"{dp}/E_int", dist["E_int"])
    if "E" in dist:
        d.floats(f"{dp}/E", dist["E"])
    for j, s in enumerate(dist.get("distribution", [])):
        sp = f"{dp}/distribution/{j}"
        for key in ("ND", "NA", "NW", "NEP", "LANG", "NL", "LTP", "NRM", "NMU"):
            if key in s:
                d.int(f"{sp}/{key}", s[key])
        if "E" in s:
            d.float(f"{sp}/E", s["E"])
        if "E'" in s:
            d.floats(f"{sp}/Eout", s["E'"])
        for key in ("A_l", "A"):
            if key in s:
                d.floats(f"{sp}/{key}", s[key])
        if "b" in s:
            for r, row in enumerate(s["b"]):
                d.floats(f"{sp}/b/{r}", row)
        if "mu_int" in s:
            d.tab2(f"{sp}/mu_int", s["mu_int"])
        for k, entry in enumerate(s.get("mu", [])):
            d.float(f"{sp}/mu/{k}/mu", entry["mu"])
            d.tab1(f"{sp}/mu/{k}/f", entry["f"])


def _pairs(d: Dump, path: str, values) -> None:
    """A list of (value, uncertainty) pairs, flattened."""
    d.floats(path, [v for pair in values for v in pair])


def dump_mf7(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    if mt == 2:
        d.int(f"{path}/LTHR", section["LTHR"])
        for i, t in enumerate(section.get("coherent", [])):
            tp = f"{path}/coherent/{i}"
            d.float(f"{tp}/T", t["T"])
            if "LT" in t:
                d.int(f"{tp}/LT", t["LT"])
                d.tab1(f"{tp}/S", t["S"])
            else:
                d.int(f"{tp}/LI", t["LI"])
                d.floats(f"{tp}/S", t["S"])
        if "incoherent" in section:
            d.float(f"{path}/incoherent/SB", section["incoherent"]["SB"])
            d.tab1(f"{path}/incoherent/W", section["incoherent"]["W"])
    elif mt == 4:
        for key in ("LAT", "LASYM", "LLN", "NI", "NS"):
            d.int(f"{path}/{key}", section[key])
        d.floats(f"{path}/B", section["B"])
        if "beta_int" in section:
            d.tab2(f"{path}/beta_int", section["beta_int"])
            d.int(f"{path}/NB", section["NB"])
        for i, block in enumerate(section.get("beta_data", [])):
            for j, t in enumerate(block):
                tp = f"{path}/beta_data/{i}/{j}"
                d.float(f"{tp}/T", t["T"])
                d.float(f"{tp}/beta", t["beta"])
                d.int(f"{tp}/LT", t["LT"])
                if j == 0:
                    d.tab1(f"{tp}/S", t["S"])
                else:
                    d.floats(f"{tp}/S", t["S"])
        for i, t in enumerate(section.get("Teff", [])):
            d.tab1(f"{path}/Teff/{i}", t)
    elif mt == 451:
        d.int(f"{path}/NA", section["NA"])
        for i, e in enumerate(section["elements"]):
            ep = f"{path}/elements/{i}"
            d.int(f"{ep}/NAS", e["NAS"])
            d.int(f"{ep}/NI", e["NI"])
            for key in ("ZAI", "LISI", "AFI", "AWRI", "SFI"):
                d.floats(f"{ep}/{key}", e[key])


def dump_mf8(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])

    if mt in (454, 459):
        d.int(f"{path}/LE", section["LE"])
        for i, y in enumerate(section["yields"]):
            yp = f"{path}/yields/{i}"
            d.float(f"{yp}/E", y["E"])
            d.int(f"{yp}/NN", y["NN"])
            d.int(f"{yp}/NFP", y["NFP"])
            # The format overloads this field; the reader keys it LE on the
            # first energy and I on the rest.
            d.int(f"{yp}/LE_or_I", y["LE"] if i == 0 else y["I"])
            for j, p in enumerate(y["products"]):
                pp = f"{yp}/products/{j}"
                d.float(f"{pp}/ZAFP", p["ZAFP"])
                d.float(f"{pp}/FPS", p["FPS"])
                d.floats(f"{pp}/Y", list(p["Y"]))
        return

    if mt == 457:
        for key in ("LIS", "LISO", "NST", "NSP"):
            d.int(f"{path}/{key}", section[key])
        d.float(f"{path}/SPI", section["SPI"])
        d.float(f"{path}/PAR", section["PAR"])
        if section["NST"] == 1:
            return
        d.floats(f"{path}/T1_2", list(section["T1/2"]))
        d.int(f"{path}/NC", section["NC"])
        _pairs(d, f"{path}/Ex", section["Ex"])
        d.int(f"{path}/NDK", section["NDK"])
        for i, m in enumerate(section["modes"]):
            mp = f"{path}/modes/{i}"
            d.float(f"{mp}/RTYP", m["RTYP"])
            d.float(f"{mp}/RFS", m["RFS"])
            d.floats(f"{mp}/Q", list(m["Q"]))
            d.floats(f"{mp}/BR", list(m["BR"]))
        for i, s in enumerate(section["spectra"]):
            sp = f"{path}/spectra/{i}"
            d.float(f"{sp}/STYP", s["STYP"])
            for key in ("LCON", "LCOV", "NER"):
                d.int(f"{sp}/{key}", s[key])
            for key in ("FD", "ER_AV", "FC"):
                d.floats(f"{sp}/{key}", list(s[key]))
            for j, r in enumerate(s.get("discrete", [])):
                rp = f"{sp}/discrete/{j}"
                d.floats(f"{rp}/ER", list(r["ER"]))
                d.float(f"{rp}/RTYP", r["RTYP"])
                d.float(f"{rp}/TYPE", r["TYPE"])
                # A discrete record is written with NT = 6 or NT = 12
                # depending on whether the internal conversion coefficients
                # were evaluated, and both lengths appear within one spectrum.
                # The Python reader slices, so an absent coefficient is an
                # empty tuple; the Rust reader makes it None. Same fact, so
                # the empty ones are skipped here rather than compared.
                for key in ("RI", "RIS", "RICC", "RICK", "RICL"):
                    if r.get(key):
                        d.floats(f"{rp}/{key}", list(r[key]))
            if "continuous" in s:
                d.float(f"{sp}/continuous/RTYP", s["continuous"]["RTYP"])
                d.tab1(f"{sp}/continuous/RP", s["continuous"]["RP"])
            if "continuous_covariance" in s:
                c = s["continuous_covariance"]
                d.int(f"{sp}/cont_cov/LB", c["LB"])
                d.floats(f"{sp}/cont_cov/Ek", c["Ek"])
                d.floats(f"{sp}/cont_cov/Fk", c["Fk"])
            if "discrete_covariance" in s:
                c = s["discrete_covariance"]
                for key in ("LS", "LB", "NE", "NERP"):
                    d.int(f"{sp}/disc_cov/{key}", c[key])
                d.floats(f"{sp}/disc_cov/Ek", c["Ek"])
                d.floats(f"{sp}/disc_cov/Fkk", c["Fkk"])
        return

    for key in ("LIS", "LISO", "NS", "NO"):
        d.int(f"{path}/{key}", section[key])
    for i, s in enumerate(section["subsections"]):
        sp = f"{path}/subsections/{i}"
        d.float(f"{sp}/ZAP", s["ZAP"])
        d.float(f"{sp}/ELFS", s["ELFS"])
        d.int(f"{sp}/LMF", s["LMF"])
        d.int(f"{sp}/LFS", s["LFS"])
        if "ND" in s:
            d.int(f"{sp}/ND", s["ND"])
            for key in ("HL", "RTYP", "ZAN", "BR", "END", "CT"):
                d.floats(f"{sp}/{key}", s[key])


def dump_mf9_mf10(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.int(f"{path}/LIS", section["LIS"])
    d.int(f"{path}/NS", section["NS"])
    for i, level in enumerate(section["levels"]):
        lp = f"{path}/levels/{i}"
        d.float(f"{lp}/QM", level["QM"])
        d.float(f"{lp}/QI", level["QI"])
        d.int(f"{lp}/IZAP", level["IZAP"])
        d.int(f"{lp}/LFS", level["LFS"])
        d.tab1(f"{lp}/func", level["Y"] if "Y" in level else level["sigma"])


def dump_mf12(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.int(f"{path}/LO", section["LO"])
    d.int(f"{path}/NK", section["NK"])
    if "Y" in section:
        d.tab1(f"{path}/Y", section["Y"])
    for i, k in enumerate(section.get("multiplicities", [])):
        kp = f"{path}/multiplicities/{i}"
        d.float(f"{kp}/Eg", k["Eg"])
        d.float(f"{kp}/ES", k["ES"])
        d.int(f"{kp}/LP", k["LP"])
        d.int(f"{kp}/LF", k["LF"])
        d.tab1(f"{kp}/y", k["y"])
    if "LG" in section:
        d.int(f"{path}/LG", section["LG"])
        d.float(f"{path}/ES_NS", section["ES_NS"])
        d.int(f"{path}/LP", section["LP"])
        d.int(f"{path}/NT", section["NT"])
        for i, t in enumerate(section["transitions"]):
            tp = f"{path}/transitions/{i}"
            d.float(f"{tp}/ES", t["ES"])
            d.float(f"{tp}/TP", t["TP"])
            if "GP" in t:
                d.float(f"{tp}/GP", t["GP"])


def dump_mf13(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.int(f"{path}/NK", section["NK"])
    if "sigma_total" in section:
        d.tab1(f"{path}/sigma_total", section["sigma_total"])
    for i, p in enumerate(section["photons"]):
        pp = f"{path}/photons/{i}"
        d.float(f"{pp}/EG", p["EG"])
        d.float(f"{pp}/ES", p["ES"])
        d.int(f"{pp}/LP", p["LP"])
        d.int(f"{pp}/LF", p["LF"])
        d.tab1(f"{pp}/sigma", p["sigma"])


def dump_mf14(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.int(f"{path}/LI", section["LI"])
    d.int(f"{path}/NK", section["NK"])
    if "LTT" in section:
        d.int(f"{path}/LTT", section["LTT"])
        d.int(f"{path}/NI", section["NI"])
    for i, s in enumerate(section.get("subsections", [])):
        sp = f"{path}/subsections/{i}"
        d.float(f"{sp}/EG", s["EG"])
        d.float(f"{sp}/ES", s["ES"])
        if "E_int" in s:
            d.tab2(f"{sp}/E_int", s["E_int"])
            d.int(f"{sp}/NE", s["NE"])
            d.floats(f"{sp}/E", s["E"])
        if "NL" in s:
            d.floats(f"{sp}/NL", s["NL"])
        for j, a in enumerate(s.get("a_lk", [])):
            d.floats(f"{sp}/a_lk/{j}", a)
        for j, p in enumerate(s.get("p_k", [])):
            d.tab1(f"{sp}/p_k/{j}", p)


def dump_mf15(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.int(f"{path}/NC", section["NC"])
    for i, s in enumerate(section["subsections"]):
        sp = f"{path}/subsections/{i}"
        d.int(f"{sp}/LF", s["LF"])
        d.tab1(f"{sp}/p", s["p"])
        d.tab2(f"{sp}/E_int", s["E_int"])
        d.int(f"{sp}/NE", s["NE"])
        d.floats(f"{sp}/E", s["E"])
        for j, g in enumerate(s["g"]):
            d.tab1(f"{sp}/g/{j}", g)


def dump_mf23(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.float(f"{path}/EPE", section["EPE"])
    d.float(f"{path}/EFL", section["EFL"])
    d.tab1(f"{path}/sigma", section["sigma"])


def dump_mf26(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.int(f"{path}/NK", section["NK"])
    for i, p in enumerate(section["products"]):
        pp = f"{path}/products/{i}"
        d.int(f"{pp}/ZAP", p["ZAP"])
        d.float(f"{pp}/AWI", p["AWI"])
        d.int(f"{pp}/LAW", p["LAW"])
        d.tab1(f"{pp}/y", p["y"])
        if "distribution" not in p:
            continue
        dist = p["distribution"]
        dp = f"{pp}/distribution"
        if "ET" in dist:
            d.tab1(f"{dp}/ET", dist["ET"])
        else:
            # LAW=1 and LAW=2 are shared with MF=6.
            dump_mf6_distribution(d, dp, dist)


def dump_mf27(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.float(f"{path}/Z", section["Z"])
    d.tab1(f"{path}/H", section["H"])


def dump_mf28(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.int(f"{path}/NSS", section["NSS"])
    for i, s in enumerate(section["shells"]):
        sp = f"{path}/shells/{i}"
        d.float(f"{sp}/SUBI", s["SUBI"])
        d.int(f"{sp}/NTR", s["NTR"])
        d.float(f"{sp}/EBI", s["EBI"])
        d.float(f"{sp}/ELN", s["ELN"])
        for key in ("SUBJ", "SUBK", "ETR", "FTR"):
            d.floats(f"{sp}/{key}", s[key])


def dump_mf33_subsection(d: Dump, sp: str, sub: dict) -> None:
    """One MF=33 subsection. Shared with MF=40, which reuses the format."""
    d.float(f"{sp}/XMF1", sub["XMF1"])
    d.float(f"{sp}/XLFS1", sub["XLFS1"])
    for key in ("MAT1", "MT1", "NC", "NI"):
        d.int(f"{sp}/{key}", sub[key])
    for i, nc in enumerate(sub["nc_subsections"]):
        np_ = f"{sp}/nc/{i}"
        d.int(f"{np_}/LTY", nc["LTY"])
        d.float(f"{np_}/E1", nc["E1"])
        d.float(f"{np_}/E2", nc["E2"])
        for key in ("NCI", "MATS", "MTS", "NEI"):
            if key in nc:
                d.int(f"{np_}/{key}", nc[key])
        for key in ("XMFS", "XLFSS"):
            if key in nc:
                d.float(f"{np_}/{key}", nc[key])
        for key in ("CI", "XMTI", "EI", "WEI"):
            if key in nc:
                d.floats(f"{np_}/{key}", nc[key])
    for i, ni in enumerate(sub["ni_subsections"]):
        ip = f"{sp}/ni/{i}"
        for key in ("LT", "LS", "LB", "NT", "NP", "NE", "NER", "NEC"):
            if key in ni:
                d.int(f"{ip}/{key}", ni[key])
        for key in ("Ek", "Fk", "El", "Fl", "Fkk", "ER", "EC", "Fkl"):
            if key in ni:
                d.floats(f"{ip}/{key}", ni[key])


def dump_mf33(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.int(f"{path}/MTL", section["MTL"])
    d.int(f"{path}/NL", section["NL"])
    for i, sub in enumerate(section["subsections"]):
        dump_mf33_subsection(d, f"{path}/subsections/{i}", sub)


def dump_mf34(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.int(f"{path}/LTT", section["LTT"])
    d.int(f"{path}/NMT1", section["NMT1"])
    # 'subsections' is always empty upstream; see issue #18. Emitting nothing
    # for it keeps the Rust side, which reproduces that, in agreement.
    for i, sub in enumerate(section["subsections"]):
        sp = f"{path}/subsections/{i}"
        for key in ("MAT1", "MT1", "NL", "NSS", "LCT"):
            d.int(f"{sp}/{key}", sub[key])
        for key in ("L", "L1", "NI"):
            d.floats(f"{sp}/{key}", sub[key])
        for j, ss in enumerate(sub["subsubsections"]):
            ssp = f"{sp}/subsubsections/{j}"
            for key in ("LS", "LB", "NT", "NE"):
                d.floats(f"{ssp}/{key}", ss[key])
            for k, values in enumerate(ss["Data"]):
                d.floats(f"{ssp}/Data/{k}", values)


def dump_mf40(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.int(f"{path}/LIS", section["LIS"])
    d.int(f"{path}/NS", section["NS"])
    for i, sub in enumerate(section["subsections"]):
        sp = f"{path}/subsections/{i}"
        d.float(f"{sp}/QM", sub["QM"])
        d.float(f"{sp}/QI", sub["QI"])
        d.int(f"{sp}/IZAP", sub["IZAP"])
        d.int(f"{sp}/LFS", sub["LFS"])
        d.int(f"{sp}/NL", sub["NL"])
        for j, ss in enumerate(sub["subsubsections"]):
            dump_mf33_subsection(d, f"{sp}/subsubsections/{j}", ss)


#: How many XSS values to record per ACE table. The array runs to hundreds of
#: thousands of numbers; a spread across it, plus both ends and every JXS entry
#: point, pins the parse without a golden file the size of the library.
ACE_XSS_SAMPLES = 2000


def ace_xss_indices(n: int, jxs) -> list[int]:
    """Which XSS indices to record. Mirrored exactly on the Rust side."""
    idx = set(range(0, n, max(1, n // ACE_XSS_SAMPLES)))
    idx.update(range(0, min(50, n)))
    idx.update(range(max(0, n - 50), n))
    # The JXS values are offsets into XSS: where a consumer actually looks.
    for j in jxs:
        if 0 <= int(j) < n:
            idx.add(int(j))
    return sorted(idx)


def dump_ace_angle(d: Dump, path: str, table) -> None:
    """Every angular distribution an ACE neutron table holds.

    LAND (JXS(8)) locates one array per reaction that emits a neutron, elastic
    scattering first; the arrays themselves sit in AND (JXS(9)). A locator of
    -1 means the angle is bound up with the energy in the DLW block instead,
    and 0 means isotropic; neither has an array to read here.
    """
    from endf.ace import TableType
    from endf.mf4 import AngleDistribution

    if table.data_type != TableType.NEUTRON_CONTINUOUS:
        return
    land, and_ = table.jxs[8], table.jxs[9]
    if land <= 0:
        return

    # NXS(5) reactions besides elastic scattering.
    n = int(table.nxs[5]) + 1
    d.int(f"{path}/n", n)
    locators = [int(table.xss[land + i]) for i in range(n)]
    d.ints(f"{path}/locators", locators)
    for i, locator in enumerate(locators):
        if locator <= 0:
            continue
        dist = AngleDistribution.from_ace(table, and_, locator)
        dump_angle_distribution(d, f"{path}/{i}", dist)


def dump_ace_dlw(d: Dump, path: str, table) -> None:
    """Every angle-energy distribution an ACE neutron table holds.

    LDLW (JXS(10)) gives one locator per reaction that emits a neutron, and
    the DLW block (JXS(11)) holds the distributions. A reaction may have
    several, chained: each entry's first word points at the next one, and the
    fourth begins the applicability of the one it introduces.
    """
    from types import SimpleNamespace

    from endf.ace import TableType
    from endf.angle_energy import AngleEnergy
    from endf.function import Tabulated1D

    if table.data_type != TableType.NEUTRON_CONTINUOUS:
        return
    ldlw, dlw = int(table.jxs[10]), int(table.jxs[11])
    if ldlw <= 0:
        return

    # Law 66 wants the reaction's Q value. Reactions are not read here, so a
    # fixed value stands in; both readers are given the same one, which leaves
    # every other field of the law under test.
    rx = SimpleNamespace(q_reaction=0.0)

    n = int(table.nxs[5])
    d.int(f"{path}/n", n)
    for i_reaction in range(1, n + 1):
        rp = f"{path}/{i_reaction}"
        lnw = int(table.xss[ldlw + i_reaction - 1])
        chain = []
        while lnw > 0:
            k = len(chain)
            chain.append(lnw)
            d.tab1(
                f"{rp}/{k}/applicability", Tabulated1D.from_ace(table, dlw + lnw + 2)
            )
            dump_angle_energy(d, f"{rp}/{k}", AngleEnergy.from_ace(table, dlw, lnw, rx))
            lnw = int(table.xss[dlw + lnw - 1])
        d.ints(f"{rp}/chain", chain)


def dump_ace_reactions(d: Dump, path: str, table) -> None:
    """Every reaction an ACE neutron table holds, index 0 being elastic."""
    from endf.ace import TableType
    from endf.reaction import Reaction

    # MTR (JXS(3)) lists the reactions; without it there are none to read.
    if table.data_type != TableType.NEUTRON_CONTINUOUS or table.jxs[3] <= 0:
        return
    n = int(table.nxs[4])
    d.int(f"{path}/n", n)
    for i_reaction in range(n + 1):
        rx = Reaction.from_ace(table, i_reaction)
        dump_reaction(d, f"{path}/{i_reaction}", rx, rx.derived_products)


def dump_incident_neutron_ace(d: Dump, path: str, table) -> None:
    """The nuclide an ACE table describes, above the reactions themselves."""
    from endf.ace import TableType
    from endf.incident_neutron import IncidentNeutron

    # ESZ (JXS(1)) is the energy grid and MTR (JXS(3)) the reaction list;
    # without them there is no nuclide to build.
    if table.data_type != TableType.NEUTRON_CONTINUOUS:
        return
    if table.jxs[1] <= 0 or table.jxs[3] <= 0:
        return
    n = IncidentNeutron.from_ace(table)

    d.text(f"{path}/name", n.name)
    d.int(f"{path}/atomic_number", n.atomic_number)
    d.int(f"{path}/mass_number", n.mass_number)
    d.int(f"{path}/metastable", n.metastable)
    d.float(f"{path}/atomic_weight_ratio", n.atomic_weight_ratio)
    d.floats(f"{path}/kTs", n.kTs)
    for i, temperature in enumerate(n.temperatures):
        d.text(f"{path}/temperatures/{i}", temperature)
        d.floats(f"{path}/energy/{temperature}", n.energy[temperature])
        if temperature in n.urr:
            d.floats(f"{path}/urr/{temperature}/energy", n.urr[temperature].energy)

    d.ints(f"{path}/mts", sorted(n.reactions))
    d.ints(
        f"{path}/redundant",
        [int(n.reactions[mt].redundant) for mt in sorted(n.reactions)],
    )
    for mt in sorted(n.reactions):
        d.ints(f"{path}/components/{mt}", n.get_reaction_components(mt))

    # The removal cross section is deliberately not dumped here. It folds the
    # elastic angular distribution into the total, and for ACE data the Python
    # `forward_fraction` returns uninitialized memory (see issue #21) so
    # there is nothing stable to compare against. It is dumped on the ENDF
    # path, where the answer is well defined.

    # The reactions `Reaction.from_ace` builds are dumped in full elsewhere.
    # What is new here are the ones this class synthesises: the total, the
    # absorption and the heating from the main energy block, and the sums it
    # builds where the table gives only the levels.
    from_ace_mts = {2}
    from_ace_mts.update(
        int(table.xss[table.jxs[3] + i - 1]) for i in range(1, int(table.nxs[4]) + 1)
    )
    for mt in sorted(set(n.reactions) - from_ace_mts):
        dump_reaction(
            d,
            f"{path}/synthesised/{mt}",
            n.reactions[mt],
            n.reactions[mt].derived_products,
        )


def dump_ace(d: Dump, path: str, table) -> None:
    d.text(f"{path}/name", table.name)
    d.float(f"{path}/atomic_weight_ratio", table.atomic_weight_ratio)
    d.float(f"{path}/kT", table.kT)
    d.float(f"{path}/temperature", table.temperature)
    d.int(f"{path}/zaid", table.zaid)
    # The suffix letter, which both sides spell the same way.
    d.text(f"{path}/data_type", table.data_type.value)
    d.ints(f"{path}/pairs_iz", [iz for iz, _ in table.pairs])
    d.floats(f"{path}/pairs_aw", [aw for _, aw in table.pairs])
    d.ints(f"{path}/nxs", table.nxs)
    d.ints(f"{path}/jxs", table.jxs)
    d.int(f"{path}/xss_len", len(table.xss))
    idx = ace_xss_indices(len(table.xss), table.jxs)
    d.ints(f"{path}/xss_idx", idx)
    d.floats(f"{path}/xss_val", [table.xss[i] for i in idx])

    dump_ace_angle(d, f"{path}/and", table)
    dump_ace_dlw(d, f"{path}/dlw", table)
    dump_ace_reactions(d, f"{path}/reaction", table)
    dump_incident_neutron_ace(d, f"{path}/nuclide", table)

    # The unresolved resonance block, when the table has one.
    from endf.urr import ProbabilityTables

    urr = ProbabilityTables.from_ace(table)
    if urr is not None:
        d.floats(f"{path}/urr/energy", urr.energy)
        d.ints(f"{path}/urr/shape", urr.table.shape)
        d.floats(f"{path}/urr/table", urr.table.ravel())
        d.int(f"{path}/urr/interpolation", urr.interpolation)
        d.int(f"{path}/urr/inelastic_flag", urr.inelastic_flag)
        d.int(f"{path}/urr/absorption_flag", urr.absorption_flag)
        d.int(f"{path}/urr/multiply_smooth", int(urr.multiply_smooth))


DUMPERS = {
    1: dump_mf1,
    2: dump_mf2,
    3: dump_mf3,
    4: dump_mf4,
    5: dump_mf5,
    6: dump_mf6,
    12: dump_mf12,
    13: dump_mf13,
    14: dump_mf14,
    15: dump_mf15,
    23: dump_mf23,
    26: dump_mf26,
    27: dump_mf27,
    28: dump_mf28,
    33: dump_mf33,
    34: dump_mf34,
    40: dump_mf40,
    7: dump_mf7,
    8: dump_mf8,
    9: dump_mf9_mf10,
    10: dump_mf9_mf10,
}


def dump(path: Path, out) -> None:
    if fixture_kind(path) == ".ace":
        dump_ace_file(path, out)
        return
    materials = endf.get_materials(path)
    source = path.relative_to(ROOT).as_posix()
    d = Dump(out)

    out.write(f"# golden reference generated from {source} by the Python reader\n")
    out.write("# regenerate with: python tools/dump_golden.py\n")
    out.write(f"SOURCE {source}\n")
    out.write(f"MATERIALS {len(materials)}\n")

    for m, material in enumerate(materials):
        out.write(f"MAT {m} {material.MAT}\n")

        # Every (MF, MT) with the number of lines its body occupies. This holds
        # the section splitter to the Python reader across the whole file,
        # including the files nothing parses yet.
        for mf, mt in sorted(material.sections):
            n_lines = len(material.section_text[mf, mt].splitlines())
            out.write(f"SECTION {m} {mf} {mt} {n_lines}\n")

        for mf, mt in sorted(material.sections):
            dumper = DUMPERS.get(mf)
            if dumper is not None:
                dumper(d, f"{m}/{mf}/{mt}", mt, material[mf, mt])

        dump_radionuclide_production(d, f"{m}/production", material)
        dump_reactions(d, f"{m}/reaction", material)
        dump_incident_neutron_endf(d, f"{m}/nuclide", material)
        dump_decay(d, f"{m}/decay", material)
        dump_incident_photon(d, f"{m}/photon", material)


def dump_incident_photon(d: Dump, path: str, material) -> None:
    """The element a photoatomic or atomic-relaxation evaluation describes.

    The Compton profiles and bremsstrahlung the Python class attaches come from
    an HDF5 file shipped with the package rather than from the evaluation, so
    they are not dumped: the Rust reader deliberately leaves that auxiliary
    data to the consumer.
    """
    from endf.incident_photon import (
        PHOTON_REACTION_NAME,
        AtomicRelaxation,
        IncidentPhoton,
    )

    has_photoatomic = any(mf == 23 for mf, _ in material.sections)
    if has_photoatomic:
        n = IncidentPhoton.from_endf(material)
        d.int(f"{path}/atomic_number", n.atomic_number)
        d.text(f"{path}/name", n.name)
        d.ints(f"{path}/mts", sorted(n.reactions))
        for mt in sorted(n.reactions):
            rx = n.reactions[mt]
            rp = f"{path}/{mt}"
            name = PHOTON_REACTION_NAME.get(mt)
            if name is not None:
                d.text(f"{rp}/name", name)
            for key in ("xs", "scattering_factor", "anomalous_real", "anomalous_imag"):
                value = getattr(rx, key)
                if value is not None:
                    d.tab1(f"{rp}/{key}", value)
            if rx.subshell_binding_energy is not None:
                d.float(f"{rp}/subshell_binding_energy", rx.subshell_binding_energy)
            if rx.fluorescence_yield is not None:
                d.float(f"{rp}/fluorescence_yield", rx.fluorescence_yield)
            d.ints(f"{rp}/components", n._get_reaction_components(mt))

    if (28, 533) in material.section_data:
        dump_atomic_relaxation(
            d, f"{path}/relaxation", AtomicRelaxation.from_endf(material)
        )


def dump_atomic_relaxation(d: Dump, path: str, relaxation) -> None:
    for i, shell in enumerate(relaxation.subshells):
        d.text(f"{path}/subshells/{i}", shell)
    for shell, value in sorted(relaxation.binding_energy.items()):
        d.float(f"{path}/binding_energy/{shell}", value)
    for shell, value in sorted(relaxation.num_electrons.items()):
        d.float(f"{path}/num_electrons/{shell}", value)
    for shell, t in sorted(relaxation.transitions.items()):
        tp = f"{path}/transitions/{shell}"
        for i, s in enumerate(t["secondary_subshell"]):
            d.text(f"{tp}/secondary/{i}", s)
        for i, s in enumerate(t["tertiary_subshell"]):
            d.text(f"{tp}/tertiary/{i}", s)
        d.floats(f"{tp}/energy", t["energy"])
        d.floats(f"{tp}/probability", t["probability"])


def dump_decay(d: Dump, path: str, material) -> None:
    """Radioactive decay data, and the fission product yields beside it."""
    from endf.decay import Decay, FissionProductYields

    if (8, 457) in material.section_text:
        dump_decay_section(d, path, Decay(material))
    if (8, 454) in material.section_text or (8, 459) in material.section_text:
        fpy = FissionProductYields(material)
        d.floats(f"{path}/fpy/energies", fpy.energies)
        for kind in ("independent", "cumulative"):
            for i, yields in enumerate(getattr(fpy, kind)):
                for j, (name, value) in enumerate(sorted(yields.items())):
                    yp = f"{path}/fpy/{kind}/{i}/{j}"
                    d.text(f"{yp}/name", name)
                    d.floats(f"{yp}/yield", [value.n, value.s])


def dump_decay_section(d: Dump, path: str, decay) -> None:
    n = decay.nuclide
    d.text(f"{path}/name", n["name"])
    for key in ("atomic_number", "mass_number", "isomeric_state", "excited_state"):
        d.int(f"{path}/{key}", n[key])
    d.float(f"{path}/mass", n["mass"])
    d.int(f"{path}/stable", int(n["stable"]))
    if n["spin"] is not None:
        d.float(f"{path}/spin", n["spin"])
    d.float(f"{path}/parity", n["parity"])

    if not n["stable"]:
        d.floats(f"{path}/half_life", [decay.half_life.n, decay.half_life.s])
        # A half-life of zero means it was not evaluated, and both readers now
        # return nothing rather than dividing by it (issue #23). Dumped as an
        # empty list rather than skipped, so the two are held to agreeing that
        # there is no decay constant, instead of the path simply being absent
        # on both sides.
        c = decay.decay_constant
        d.floats(f"{path}/decay_constant", [] if c is None else [c.n, c.s])
    e = decay.decay_energy
    d.floats(f"{path}/decay_energy", [e.n, e.s])
    for key, value in sorted(decay.average_energies.items()):
        d.floats(f"{path}/average_energies/{key}", [value.n, value.s])

    for i, mode in enumerate(decay.modes):
        mp = f"{path}/modes/{i}"
        d.text(f"{mp}/parent", mode.parent)
        d.text(f"{mp}/modes", ",".join(mode.modes))
        d.text(f"{mp}/daughter", mode.daughter)
        d.floats(f"{mp}/energy", [mode.energy.n, mode.energy.s])
        d.floats(
            f"{mp}/branching_ratio", [mode.branching_ratio.n, mode.branching_ratio.s]
        )

    for radiation, spectrum in sorted(decay.spectra.items()):
        sp = f"{path}/spectra/{radiation}"
        d.text(f"{sp}/continuous_flag", spectrum["continuous_flag"])
        for key in (
            "discrete_normalization",
            "energy_average",
            "continuous_normalization",
        ):
            d.floats(f"{sp}/{key}", [spectrum[key].n, spectrum[key].s])
        for j, line in enumerate(spectrum.get("discrete", [])):
            lp = f"{sp}/discrete/{j}"
            d.floats(f"{lp}/energy", [line["energy"].n, line["energy"].s])
            d.text(f"{lp}/from_mode", ",".join(line["from_mode"]))
            if line["type"] is not None:
                d.text(f"{lp}/type", line["type"])
            d.floats(f"{lp}/intensity", [line["intensity"].n, line["intensity"].s])
            for key in (
                "positron_intensity",
                "internal_pair",
                "total_internal_conversion",
                "k_shell_conversion",
                "l_shell_conversion",
            ):
                if key in line:
                    d.floats(f"{lp}/{key}", [line[key].n, line[key].s])
        if "continuous" in spectrum:
            d.text(
                f"{sp}/continuous_from_mode", ",".join(spectrum["continuous"]["type"])
            )
            d.tab1(f"{sp}/continuous", spectrum["continuous"]["probability"])

    if n["stable"] or decay.half_life.n != 0.0:
        for particle, dist in sorted(decay.sources.items()):
            dump_univariate(d, f"{path}/sources/{particle}", dist)


def dump_incident_neutron_endf(d: Dump, path: str, material) -> None:
    """The nuclide an ENDF evaluation describes.

    The reactions themselves are dumped by `dump_reactions`; what is checked
    here is the identity the class derives and which reactions it collects.
    """
    from endf.incident_neutron import IncidentNeutron

    if (1, 451) not in material:
        return
    n = IncidentNeutron.from_endf(material)
    d.text(f"{path}/name", n.name)
    d.int(f"{path}/atomic_number", n.atomic_number)
    d.int(f"{path}/mass_number", n.mass_number)
    d.int(f"{path}/metastable", n.metastable)
    d.text(f"{path}/atomic_symbol", n.atomic_symbol)
    d.ints(f"{path}/mts", sorted(n.reactions))
    for mt in sorted(n.reactions):
        d.ints(f"{path}/components/{mt}", n.get_reaction_components(mt))

    # The removal cross section, which folds the elastic angular distribution
    # into the total. Several cutoffs, since each picks a different slice of
    # the forward cone.
    if 1 in n.reactions and 2 in n.reactions:
        for cutoff in (-1.0, 0.0, 0.5):
            name = f"{cutoff:+.1f}"
            d.tab1(f"{path}/removal_xs/{name}", n.removal_xs("0K", cutoff))


def dump_reactions(d: Dump, path: str, material) -> None:
    """Every reaction the material evaluates, gathered from its files."""
    from endf.reaction import FISSION_MTS, Reaction, _get_fission_products_endf

    mts = sorted(mt for mf, mt in material.sections if mf == 3)
    d.ints(f"{path}/mts", mts)
    for mt in mts:
        rx = Reaction.from_endf(mt, material)
        # `Reaction.from_endf` computes the derived products and then drops
        # them; there is a TODO in the source saying as much. The Rust
        # reaction keeps them, so they are taken from the helper here and
        # compared rather than left out of the check.
        derived = []
        if mt in FISSION_MTS:
            derived = _get_fission_products_endf(material, mt)[1]
        dump_reaction(d, f"{path}/{mt}", rx, derived)


def dump_reaction(d: Dump, path: str, rx, derived_products) -> None:
    d.int(f"{path}/MT", rx.MT)
    d.float(f"{path}/q_reaction", rx.q_reaction)
    d.float(f"{path}/q_massdiff", rx.q_massdiff)
    d.int(f"{path}/redundant", int(rx.redundant))
    d.int(f"{path}/center_of_mass", int(rx.center_of_mass))
    for temperature, xs in sorted(rx.xs.items()):
        d.tab1(f"{path}/xs/{temperature}", xs)
    for kind, products in (
        ("products", rx.products),
        ("derived_products", derived_products),
    ):
        d.int(f"{path}/n_{kind}", len(products))
        for i, product in enumerate(products):
            dump_product(d, f"{path}/{kind}/{i}", product)


def dump_product(d: Dump, path: str, product) -> None:
    from numpy.polynomial import Polynomial

    from endf.function import Tabulated1D

    d.text(f"{path}/name", product.name)
    d.text(f"{path}/emission_mode", product.emission_mode)
    d.float(f"{path}/decay_rate", product.decay_rate)
    if isinstance(product.yield_, Tabulated1D):
        d.text(f"{path}/yield/kind", "tabulated")
        d.tab1(f"{path}/yield/f", product.yield_)
    elif isinstance(product.yield_, Polynomial):
        d.text(f"{path}/yield/kind", "polynomial")
        d.floats(f"{path}/yield/coef", product.yield_.coef)
    else:
        raise TypeError(f"unexpected yield {type(product.yield_)}")
    for i, applicability in enumerate(product.applicability):
        d.tab1(f"{path}/applicability/{i}", applicability)
    d.int(f"{path}/n_distribution", len(product.distribution))
    for i, dist in enumerate(product.distribution):
        dump_angle_energy(d, f"{path}/distribution/{i}", dist)


def dump_radionuclide_production(d: Dump, path: str, material) -> None:
    """The MF=8/9/10 join, which is a derived view rather than a section."""
    from endf.radionuclide_production import radionuclide_production

    production = radionuclide_production(material)
    d.ints(f"{path}/mts", sorted(production))
    for mt, states in sorted(production.items()):
        for i, state in enumerate(states):
            sp = f"{path}/{mt}/{i}"
            d.int(f"{sp}/ZAP", state.ZAP)
            d.int(f"{sp}/LFS", state.LFS)
            d.float(f"{sp}/QM", state.QM)
            d.float(f"{sp}/QI", state.QI)
            if state.ELFS is not None:
                d.float(f"{sp}/ELFS", state.ELFS)
            d.float(f"{sp}/excitation_energy", state.excitation_energy)
            if state.yields is not None:
                d.tab1(f"{sp}/yields", state.yields)
            if state.cross_section is not None:
                d.tab1(f"{sp}/cross_section", state.cross_section)


def dump_ace_file(path: Path, out) -> None:
    """An ACE fixture, which has tables rather than materials."""
    import endf.ace

    tables = endf.ace.get_tables(path)
    source = path.relative_to(ROOT).as_posix()
    d = Dump(out)

    out.write(f"# golden reference generated from {source} by the Python reader\n")
    out.write("# regenerate with: python tools/dump_golden.py\n")
    out.write("KIND ace\n")
    out.write(f"SOURCE {source}\n")
    out.write(f"TABLES {len(tables)}\n")
    for i, table in enumerate(tables):
        dump_ace(d, str(i), table)


def fixtures() -> list[Path]:
    found = []
    for directory in FIXTURE_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if fixture_kind(path) in FIXTURE_SUFFIXES:
                found.append(path)
    return found


def golden_for(path: Path) -> Path:
    """Where a fixture's dump is stored."""
    # `Path.stem` strips only one suffix, so a fixture named
    # `n-095_Am_244.endf.xz` needs both taken off.
    stem = Path(path.stem).stem if path.suffix == ".xz" else path.stem
    return GOLDEN_DIR / f"{stem}.txt.xz"


#: Relative tolerance for the paths below. Everything else is compared as text,
#: which is exact: the dump writes the shortest round-tripping decimal.
TOLERANCE = 1e-12


def is_derived(path: str) -> bool:
    """Whether a path holds arithmetic rather than a parsed value.

    These are the same paths `crates/endf/tests/golden.rs` compares with a
    tolerance, and for the same reason: they are computed, and the computation
    moves in the last bit or two between NumPy releases. The forward-scattered
    fraction integrates a Legendre series, whose Clenshaw recurrence NumPy
    re-associated between 2.2 and 2.4, and the removal cross section folds that
    into the total.
    """
    return (
        path.endswith("/evaly")
        or ("/reaction/" in path and "/yield/f/" in path)
        or path.endswith("/decay_constant")
        or path.endswith("/decay_energy")
        or "/forward_fraction/" in path
        or "/removal_xs/" in path
    )


def same_line(stored: str, fresh: str) -> bool:
    """Whether two dump lines say the same thing."""
    if stored == fresh:
        return True
    a, b = stored.split(), fresh.split()
    # Only float values are ever compared loosely, and only on derived paths.
    if len(a) != len(b) or a[:3] != b[:3] or a[0] != "V" or a[2] != "F":
        return False
    if not is_derived(a[1]):
        return False
    for x, y in zip(a[3:], b[3:]):
        u, v = float(x), float(y)
        if u == v:
            continue
        if u != u and v != v:  # both NaN
            continue
        if abs(u - v) > TOLERANCE * max(abs(u), abs(v)):
            return False
    return True


def check(paths: list[Path]) -> int:
    """Compare the stored dumps against freshly generated ones.

    Text, not bytes: the file is xz-compressed and two encoders can write the
    same content differently, so a byte comparison would fail on a machine
    whose liblzma is not the one that wrote the file.
    """
    drifted = 0
    for path in paths:
        target = golden_for(path)
        if not target.exists():
            print(f"{target.relative_to(ROOT)}: missing", file=sys.stderr)
            drifted += 1
            continue

        fresh = io.StringIO()
        dump(path, fresh)
        with lzma.open(target, "rt") as fh:
            stored = fh.read()

        stored_lines = stored.splitlines()
        fresh_lines = fresh.getvalue().splitlines()
        if len(stored_lines) == len(fresh_lines) and all(
            same_line(a, b) for a, b in zip(stored_lines, fresh_lines)
        ):
            continue

        drifted += 1
        print(f"{target.relative_to(ROOT)}: drifted", file=sys.stderr)
        diff = difflib.unified_diff(
            stored_lines,
            fresh_lines,
            fromfile="stored",
            tofile="regenerated",
            lineterm="",
            n=1,
        )
        # Enough context to identify the value, not the whole file.
        for i, line in enumerate(diff):
            if i >= 40:
                print("  ...", file=sys.stderr)
                break
            print(f"  {line[:400]}", file=sys.stderr)
    return drifted


def main() -> None:
    args = sys.argv[1:]
    checking = "--check" in args
    args = [a for a in args if a != "--check"]

    paths = [Path(a).resolve() for a in args] or fixtures()
    if not paths:
        sys.exit("no ENDF fixtures found; pass one explicitly")

    if checking:
        drifted = check(paths)
        if drifted:
            sys.exit(f"{drifted} golden file(s) no longer match the reader")
        print(f"{len(paths)} golden files match", file=sys.stderr)
        return

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = golden_for(path)
        # Built whole in memory and written only once it is complete. Writing
        # straight to the file leaves a truncated golden behind when the dump
        # raises part way, and a truncated golden is worse than none: the Rust
        # harness compares the two maps whole, so it reports every path after
        # the failure as missing and buries the actual error.
        buffer = io.StringIO()
        dump(path, buffer)
        # The dumps are as repetitive as the evaluations they come from and
        # compress about seven to one, so they are stored the same way.
        with lzma.open(target, "wt", preset=9) as out:
            out.write(buffer.getvalue())
        print(
            f"{target.relative_to(ROOT)}  <-  {path.relative_to(ROOT)}", file=sys.stderr
        )


if __name__ == "__main__":
    main()
