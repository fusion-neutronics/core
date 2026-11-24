# Validation

YANI is run against every open, reproducible benchmark we have found, and the
results, the scripts that produced them and a figure for every single case are
published separately:

**[The verification and validation site](https://fusion-neutronics.github.io/yani-verification-and-validation/)**,
and [the repository](https://github.com/fusion-neutronics/yani-verification-and-validation)
it is built from.

It is a separate repository because it is a different kind of thing. It holds
gigabytes of result JSON and hundreds of figures, it takes hours of solving to
regenerate, and it should be runnable by someone who has never cloned this one.
Everything there reproduces from a clean checkout with the benchmark data
committed, so nothing is fetched from the IAEA to check a number.

## What is covered

| benchmark | what is measured | cases | data behind another code's published results |
| --- | --- | ---: | --- |
| [FNS decay heat](https://fusion-neutronics.github.io/yani-verification-and-validation/benchmarks/fns-decay-heat/) | specific decay heat after a 14 MeV irradiation | 132 | TENDL-2017 |
| [Effective cross sections](https://fusion-neutronics.github.io/yani-verification-and-validation/benchmarks/effective-cross-sections/) | spectrum-averaged cross sections in 23 neutron fields | 365 | TENDL-2017 |

**FNS decay heat.** JAEA's Fusion Neutronics Source held a 1 g foil in a 14 MeV
field of about 1.1e10 n/cm²/s, withdrew it, and measured the heat its activation
products gave off as it cooled. 73 foils across three campaigns: two five minute
irradiations followed for about an hour, and one 7.6 hour irradiation followed
out to 400 days.

**Effective cross sections.** A foil is irradiated in a known field and the
product counted, giving the production rate per target atom per unit flux. That
is one fold of a cross section against a spectrum with nothing else in the way,
which makes it the most direct available test of the step YANI takes before any
solving happens. 89 of the 365 reactions measure an isomeric state directly, so
for those the branching fraction is not a correction to the answer, it is the
answer. See [isomeric branching](usage.md#weighted-from-a-solve).

Both come from the IAEA's [CoNDERC](https://nds.iaea.org/conderc/) collection.
What makes these archives worth building on is that each carries three things
rather than one: the measurement with its experimental uncertainty, the
conditions needed to reproduce it (the neutron field as group fluxes, and for
the decay heat cases the foil composition and mass and the schedule), and
another code's calculated results for the same cases. So a disagreement can be
attributed rather than merely noted.

## How to read a comparison there

**The like-for-like line is a verification result.** The other code's published
results used TENDL-2017 cross sections and UKAEA's `decay_2012` decay library,
so that is what YANI is run with on that line. Two independently written codes
given identical evaluations should land on the same answer, and when they do,
that is evidence both implementations are right. A large win on identical data
would be evidence that one of them is wrong, and it should be read that way
rather than celebrated.

**The other lines are a different question.** YANI can be pointed at TENDL-2025,
JEFF-4.0, ENDF/B-VIII.1, FENDL-3.2d or a mixture assembled subsection by
subsection, and where one of those tracks the measurement better, that is a
statement about data rather than about code. Both are worth knowing, they are
not the same claim, and every page keeps them apart.

**Where a deviation is shared, it belongs to neither code.** Several foils sit
tens or hundreds of percent from the measurement in both codes at once. That is
the evaluation, and no amount of work on either code will move it.

## What is not covered

Stated here so it cannot be mistaken for coverage that exists.

- **Every field in both benchmarks is a DT field**, at JAEA-FNS, FNG-Frascati or
  TU Dresden. There is no fission-spectrum or thermal validation on the site yet.
- **No foil in either set is fissionable**, so the fission-yield path these
  benchmarks exercise is none of it, however the yield settings are configured.
- **Self-shielding** is compared against measurement on a single case.
- **Displacement damage (dpa)** is not compared against measurement or against
  another code on the site yet.
- Coverage of the nuclear-data uncertainty machinery is reported per run by
  `get_data_uncertainty_info`, and the site's methodology page states what the
  benchmark runs there did and did not propagate. See
  [Nuclear-data uncertainty](usage.md#nuclear-data-uncertainty).

If you know of an open, reproducible benchmark that is not covered,
[raise an issue](https://github.com/fusion-neutronics/yani-verification-and-validation/issues)
and it will be added with a reproducible script.

## Reproducing it

```bash
git clone https://github.com/fusion-neutronics/yani-verification-and-validation
cd yani-verification-and-validation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python benchmarks/fns_decay_heat/run.py --case Fe
```

`python tools/run_all.py --plan` says what a full run would do and roughly how
long it would take, and checks that YANI is installed, the benchmark data is
present and the disk can hold the nuclear data a full run caches, before running
any of it.
