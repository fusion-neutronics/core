# CRAM order: what order 50 buys over order 48

Short answer: nothing measurable, and it costs a 25th linear solve per step.
CRAM-48 stays the default. The numbers are below so the decision can be
revisited rather than remembered.

## What was compared

`cram50` and `cram50_sparse` were added beside `cram48`, built from the order-50
Remez fit published by [ojschumann/CRAM-Coefficients][repo] (MIT) using
`tools/gen_cram_coefficients.py`. Order 50 is 25 conjugate pairs against order
48's 24.

That script is checked by running it on the same repository's order-48 fit,
where it returns CRAM48's `alpha0` exactly and CRAM48's `theta` set to 3.8e-16
relative. Its `alpha` values differ from Pusa's published ones and are meant to:
which numerator zero is paired with which pole is a free choice that changes
each residue but not their product. Evaluated as a solver the two pairings agree
to 6.5e-16 on an 8-state operator and 1.4e-15 on a 64-state one.

[repo]: https://github.com/ojschumann/CRAM-Coefficients

## Order 50 is a genuinely better approximation

Each fit equioscillates, so its `alpha0` is also its sup-norm distance from
`exp` on the negative real axis:

| order | sup norm \|r(x) - exp(x)\| |
|---|---|
| 48 | 2.258e-47 |
| 50 | 2.617e-49 |

Order 50 is about 86 times closer. In exact arithmetic it is the better
approximation, and by a wide margin.

## None of it is reachable in double precision

Both figures are roughly thirty decades below anything an `f64` can express
relative to an O(1) answer, so neither approximation error is the thing being
measured when the solver runs.

What is being measured is the dynamic range of the recurrence. The incomplete
partial fraction form accumulates through values of order `1/alpha0` before the
final scaling brings the answer back to O(1):

| order | `alpha0` | peak accumulator | implied relative floor |
|---|---|---|---|
| 48 | 2.258e-47 | 4.385e+46 | 2.198e-16 |
| 50 | 2.617e-49 | 3.783e+48 | 2.198e-16 |

A relative perturbation of one machine epsilon anywhere in that climb survives
into the answer at the same relative size. Both orders land on the same floor,
one machine epsilon, and a better rational approximation cannot get underneath
it.

Measured end to end against `scipy.linalg.expm` on random decay-like operators,
over matrix sizes 8 / 64 / 256, decay-constant ranges of 6 / 10 / 14 decades and
steps of 1e3 s and 1e7 s, 18 cases in total:

- order 48 better in **3** cases (all at n=8, where it is 7.7e-15 against
  1.0e-14, 1.5e-14 against 1.8e-14 and 7.8e-16 against 1.7e-15)
- **tie in 15**
- order 50 better in **0**

The scalar approximation evaluated in 200-bit arithmetic tells the same story
from the other direction: with `f64` coefficients, order 48 reaches 7.1e-16 and
order 50 reaches 1.6e-15, the order-50 result being slightly worse for exactly
the reason above, two extra decades of accumulator to round through.

## The cost

Order 50 is 25 linear solves per step against 24, so about 4% more work in the
part of a step that dominates it, for no accuracy.

## The other reason to keep order 48

CRAM-48 with Pusa's published coefficients is what OpenMC's `CRAM48` uses, and
what the inventory codes the V&V repository compares against use. Keeping it
means a disagreement with another code can be attributed to data or to chain
assembly rather than to the two sides evaluating different rational functions.
That is worth more than 4% of a step.

## Reproducing this

```bash
pip install 'mpmath==1.3.0' numpy scipy
curl -LO https://raw.githubusercontent.com/ojschumann/CRAM-Coefficients/master/result/R048.dat
curl -LO https://raw.githubusercontent.com/ojschumann/CRAM-Coefficients/master/result/R050.dat

python tools/gen_cram_coefficients.py R048.dat          # checks the script
python tools/gen_cram_coefficients.py R050.dat --rust   # the arrays in cram.rs
```

`cargo test -p yani --lib cram` covers the two orders agreeing on a stiff
three-nuclide chain across four step lengths, and the sparse path agreeing with
the dense one at order 50.

Both comparisons scale by the largest population in the vector rather than
per-component. A component many decades below the largest is round-off in both
solvers and holds no significant figures to compare, so dividing by it compares
two noise values.
