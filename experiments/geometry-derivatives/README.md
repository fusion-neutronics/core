# Geometry derivatives: prototype

Experimental work for #574 (differentiable CSG and implicit geometry). Not part
of the workspace and not used by yamc; kept so the experiment is not lost.

A standalone program that runs one-group transport (isotropic scattering, a
track-length reaction-rate tally) on the real `yamc-geo` surfaces and estimates
the derivative of the tally with respect to surface parameters in the same run.
Each case is compared with a central finite difference of separate runs.

## The estimator

When a particle of weight `w` crosses a differentiated interface from material
A into B, with `delta = ds/dtheta = -f_theta / (Omega . grad f)`:

- direct: add `w delta (R_A - R_B)` to the tally's derivative;
- pass: the particle's derivative weight grows by `w delta (St_B - St_A)`;
- branch: bank two derivative-only particles forced to collide at the crossing,
  one in A with weight `+w delta St_A`, one in B with weight `-w delta St_B`.

## Running

```
cargo run --release                      # every case
cargo run --release -- tokamak gyroid    # named cases
SCALE=0.1 cargo run --release            # fewer histories
```

Cases: `absorber`, `spheres`, `tokamak`, `stellarator`, `tokamak-shape`,
`stellarator-shape`, `gyroid`. `NB` (copies of each branch) and `DMAX` (cap on
`delta`, biased) are the two variance diagnostics.

## Results

`results.txt` holds the runs reported in #574 (against `0e4834f5`). `z` is the
difference between the estimator and the finite difference in combined standard
errors. Summary:

- every comparison agrees within 1.9 standard errors, the analytic case to 0.1%;
- stellarator Fourier coefficients are only confirmed at 30 to 50% precision;
- at equal run time the estimator beats finite differences for the breeder
  thickness, ties for the wall thickness and loses for elongation and for rigid
  shifts of a thin wall: variance reduction is the open problem.

## Limits

One-group physics only, the source is held fixed when the plasma shape moves,
surface tracking only, and the gradients of implicit formulas are differenced
rather than read from the tape's duals.
