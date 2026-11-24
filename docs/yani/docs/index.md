# YANI

**YANI** (Yet Another Nuclide Inventory) computes transmutation and activation.
A material, an irradiation schedule and a neutron spectrum go in; inventories,
activities, decay heat, decay photon spectra and displacement damage (dpa) come
out.

```bash
pip install yani
```

!!! tip "Try it without installing anything"

    [**yani-online**](https://fusion-neutronics.github.io/yani-online/) runs
    YANI in the browser: build a material, build an irradiation and cooling
    schedule, run a real transmutation, and read back inventory, activity,
    decay heat, contact dose rate and decay photon lines.

    It is the same Rust solver compiled to WebAssembly, against the same
    continuous-energy cross sections, so it answers what the wheel answers
    rather than a reduced version of it. The solve runs on your machine and
    nothing is uploaded. The page will write your setup out as a Python script
    when you want to carry it over here.

## What it solves

The nuclide inventory of a material under irradiation is a system of coupled
linear ODEs, the Bateman equations with reaction terms:

$$
\frac{\mathrm{d}N_i}{\mathrm{d}t} =
\underbrace{\sum_{j}\left(\lambda_{j\rightarrow i} + \sigma_{j\rightarrow i}\,\phi\right)N_j}_{\text{production}}
\;-\;
\underbrace{\left(\lambda_i + \sigma_i\,\phi\right)N_i}_{\text{loss}}
$$

where $N_i$ is the atom density of nuclide $i$, $\lambda_{j\rightarrow i}$ the
decay constant of each parent $j$ that decays into it, $\sigma_{j\rightarrow i}$
the microscopic cross section of each reaction on $j$ that produces it, and
$\phi$ the scalar flux. Every nuclide is produced by both routes, decay and
neutron reaction, and lost by both.

Assembled over every nuclide in the network this is a matrix $A$, and one
timestep is a matrix exponential:

$$
\mathbf{N}(t) = e^{At}\,\mathbf{N}(0)
$$

which YANI evaluates with CRAM (the Chebyshev Rational Approximation Method) at
order 48, following [Pusa 2010][pusa2010] and [Pusa 2015][pusa2015], using a
sparse LU factorization whose symbolic factorization is reused across the poles.
Reaction rates are held at their beginning-of-step values, so a schedule of $n$
steps is $n$ matrix exponentials per material.

The coefficients of $A$ come from a transmutation network (decay constants,
branching, reaction products, fission yields) and from folding the spectrum
against continuous-energy cross sections to get each $\sigma\phi$.

[Nuclear data libraries](libraries.md) describes the six available libraries
and what each contains. [Method](method.md) describes how the network, the
collapse, self-shielding, schedules, derived quantities (activity, decay heat,
photon emission, contact dose, clearance, displacement damage) and uncertainty
propagation work.

## Whether the answers are right

YANI is run against 497 measured cases from two IAEA CoNDERC archives, both of
which publish another inventory code's results beside the measurement, so a
disagreement can be attributed rather than merely noted. Every case has a
figure, every number has the script that produced it, and what is *not* covered
is listed too. See [Validation](validation.md).

[pusa2010]: https://doi.org/10.13182/NSE09-14
[pusa2015]: https://doi.org/10.13182/NSE15-26
