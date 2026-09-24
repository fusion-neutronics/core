#!/usr/bin/env python3
"""Derive CRAM coefficients from a published Remez fit, for `crates/yani/src/cram.rs`.

The coefficients in `cram.rs` are the incomplete partial fraction form,

    r(x) = alpha0 * prod_j ( 1 + 2 Re( alpha_j / (x - theta_j) ) )

which is what the solver's accumulate-and-scale recurrence evaluates, and what
OpenMC's `CRAM48` evaluates too. They are derived here from the order-N Remez
fits published by

    https://github.com/ojschumann/CRAM-Coefficients   (MIT)

which stores each fit as a pickle of an `mpmath` multiprecision `Remez` object
holding the numerator and denominator polynomials p and q. `theta` are the
roots of q; `alpha` are the residues of each conjugate-pair factor, Pusa 2016
eq. 20; `alpha0` is `p_N / q_N`, which for an equioscillating fit is also the
sup-norm error against `exp` on the negative real axis.

Run it on the order-48 fit first, which is how its output on any other order is
checked. It returns CRAM48's `alpha0` exactly and CRAM48's `theta` set to
3.8e-16 relative. The `alpha` values differ from the published ones, and are
meant to: which numerator zero is paired with which pole is a free choice that
changes each residue but not their product, and Pusa's table made a different
choice from the heuristic below. As a solver the two agree to ~1e-15.

    python tools/gen_cram_coefficients.py R048.dat --rust
    python tools/gen_cram_coefficients.py R050.dat --rust

Needs `mpmath==1.3.0` (the pickles predate mpmath 1.4's module layout) and
numpy. The pairing of p's roots to theta values is a free choice that changes the
individual alphas but not their product; this follows the upstream script's
heuristic, so its output matches that script rather than Pusa's table.
"""

import argparse
import pickle
import sys

import mpmath
from numpy.polynomial import Polynomial

mpmath.mp.prec = 1024

# The fits are third-party pickles, so only the classes they legitimately need
# can be constructed. Everything else raises rather than executing.
ALLOWED = {
    ("remez", "Remez"),
    ("mpmath.ctx_mp_python", "mpf"),
    ("mpmath.ctx_mp_python", "mpc"),
    ("numpy.polynomial.polynomial", "Polynomial"),
    ("numpy.polynomial._polybase", "ABCPolyBase"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
}


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to construct {module}.{name}")


def _load(path):
    with open(path, "rb") as fh:
        return _Restricted(fh).load()


def _find_roots(poly, z=1j):
    """Newton root finding, deflating each root as it is found."""
    roots = []
    deflated = Polynomial(1)
    while deflated.degree() < poly.degree():
        dp = poly.deriv()
        dq = deflated.deriv()
        while True:
            f = poly(z) / deflated(z)
            df = (dp(z) - poly(z) * dq(z) / deflated(z)) / deflated(z)
            dz = -f / df
            z += dz
            if abs(dz) < mpmath.mpf("1e-250"):
                break
        if abs(z.imag) < mpmath.mpf("1e-250"):
            deflated = deflated * Polynomial((-z.real, 1))
            roots.append(mpmath.mpc(z.real, 0))
        else:
            deflated = deflated * Polynomial((-z, 1)) * Polynomial((-z.conjugate(), 1))
            roots.append(z)
        z += 1j
    return roots


def _residue(theta, t1, t2):
    """Residue of one conjugate-pair factor. Pusa 2016 eq. 20."""
    return -0.5j * (theta - t1) * (theta - t2) / theta.imag


def coefficients(path):
    fit = _load(path)
    # The pickles predate numpy 1.24, so their Polynomial instances carry no
    # `_symbol` and every derived operation raises. Rebuilding from the
    # coefficient arrays is lossless: the coefficients are the object.
    p = Polynomial(fit.p.coef)
    q = Polynomial(fit.q.coef)
    alpha0 = p.coef[-1] / q.coef[-1]

    theta = _find_roots(q)
    theta.sort(key=lambda z: abs(z.imag))

    zeros = _find_roots(p)
    zeros.sort(key=lambda z: -z.real)
    complex_zeros = [z for z in zeros if abs(z.imag) > mpmath.mpf("1e-150")]
    real_zeros = [z.real for z in zeros if abs(z.imag) <= mpmath.mpf("1e-150")]
    if 2 * len(complex_zeros) + len(real_zeros) != 2 * len(theta):
        raise SystemExit(f"{path}: root count does not match the polynomial degree")
    if len(real_zeros) % 2:
        raise SystemExit(f"{path}: odd number of real zeros, cannot pair them")

    # Pair the largest real zero with the smallest, and so on, so that no very
    # large magnitude is combined with another. Then order by magnitude so a
    # small zero meets a theta with a small imaginary part, which is the
    # denominator of the residue.
    real_zeros.sort(key=abs)
    last = len(real_zeros) - 1
    paired = [(real_zeros[i] + 1j * real_zeros[last - i], False)
              for i in range(len(real_zeros) // 2)]
    paired += [(z, True) for z in complex_zeros]
    paired.sort(key=lambda kv: abs(kv[0]))

    alpha = []
    for i, (z, already_complex) in enumerate(paired):
        if already_complex:
            alpha.append(_residue(theta[i], z, z.conjugate()))
        else:
            alpha.append(_residue(theta[i], z.real, z.imag))

    return fit.N, alpha0, theta, alpha


def _rust(v):
    return f"{float(v):.15e}".replace("e-0", "e-").replace("e+0", "e+")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fit", help="an R0NN.dat from the upstream repository's result/ directory")
    ap.add_argument("--rust", action="store_true", help="emit Rust arrays for cram.rs")
    args = ap.parse_args()

    order, alpha0, theta, alpha = coefficients(args.fit)
    pairs = len(theta)

    if not args.rust:
        print(f"order  = {order}")
        print(f"alpha0 = {float(alpha0):.16e}   (also the sup-norm error against exp)")
        for i, (t, a) in enumerate(zip(theta, alpha)):
            print(f"[{i:2}] theta = {float(t.real):+.15e} {float(t.imag):+.15e}"
                  f"  alpha = {float(a.real):+.15e} {float(a.imag):+.15e}")
        return

    print(f"#[allow(clippy::excessive_precision)]")
    print(f"const CRAM{order}_ALPHA: [Complex64; {pairs}] = [")
    for a in alpha:
        print(f"    Complex64::new({_rust(a.real)}, {_rust(a.imag)}),")
    print("];")
    print()
    print(f"#[allow(clippy::excessive_precision)]")
    print(f"const CRAM{order}_THETA: [Complex64; {pairs}] = [")
    for t in theta:
        print(f"    Complex64::new({_rust(t.real)}, {_rust(t.imag)}),")
    print("];")
    print()
    print(f"const CRAM{order}_ALPHA0: f64 = {float(alpha0):.15e};")


if __name__ == "__main__":
    sys.exit(main())
