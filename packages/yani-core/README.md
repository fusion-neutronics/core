# yani-core

Yet Another Nuclide Inventory: transmutation and activation without transport. A
material, an irradiation schedule and a neutron spectrum in; inventories,
activities, decay heat and hydrogen and helium gas production out.

This is the compiled distribution, and it provides the `yani` module itself, so
the import name and the distribution name differ (as `pillow` provides `PIL`).
Install [`yani`](https://pypi.org/project/yani/) rather than this: it pins the
matching version of this wheel and is where the documentation lives.

```python
import yani

yani.cross_section_data = "endf-b8.1"

steel = yani.materials.pnnl.material("Steel, Stainless 316", volume=1000.0)
spectrum = yani.NeutronSource(
    energy=yani.sources.Histogram([1e-5, 1e5, 1e6, 1.5e7], [1e12, 1e13, 1e14])
)
schedule = yani.PulseSchedule([
    yani.Pulse(rate=1.11e14, duration=(1, "a"), source=spectrum),
    yani.Cooldown(duration=(1, "d")),
])

results = steel.transmute(schedule=schedule)
final = results.get_final_material(steel.id or 0)
print(final.activity(), "Bq")
print(final.decay_heat(), "W")
print(final.contact_dose(), "Gy/h")
print(final.clearance_index("UK_EPR16_out_of_scope").index)
print(results.get_gas_production(steel.id or 0)["He"], "appm He")
```

Displacement damage comes from the same call: pass `displacement_damage=True`
for the damage energy per atom and NRT dpa over the schedule, per element and
for the material.

```python
tungsten = yani.Material({"W": 1.0}, density=19.3)
results = tungsten.transmute(schedule=schedule, displacement_damage=True)
results.get_dpa(0)  # cumulative, one value per state
results.get_dpa(0, element="W")
results.get_damage_energy(0)  # eV per atom, the model-free input
results.get_displacement_damage_info(0)["displacement_energies"]
```

The displacement threshold energies default to ASTM E521, and to the OECD-NEA
2015 report "Primary Radiation Damage in Materials" (NEA/NSC/DOC(2015)9) for
elements E521 does not cover; override any of them with
`displacement_energies={"Fe": 40.0}`. An element in neither source must be
given, and is never guessed.

## Relationship to yamc

`yamc` is a superset: it ships
everything here plus neutron and photon transport, geometry, tallies and
transport-coupled transmutation. The two are **alternatives**. Installing both
puts two extension modules in one process, so `yamc.Material` and
`yani.Material` are distinct types and the nuclear-data configuration
(`cross_section_data`, the four `transmutation_*` sources) exists twice, once
per package.

Pick `yani` when you have a spectrum already and want none of the transport
stack; pick `yamc` when the spectrum should come from a transport solve.

## What it does not do

- One stepper (`ForwardEulerStepper`, beginning-of-step rates). No
  predictor-corrector.
- Uncertainty is by resampling: pass `data_uncertainty=yani.DataUncertainty()`
  to `transmute` for a standard deviation on inventories, activity, decay heat,
  dose and gas production. It perturbs MF=33 cross sections, MF=32 resonance
  parameters (drawn per replica and the resonance cross sections rebuilt from
  them, where the library publishes them, and otherwise their first-order
  group covariance), half-lives, decay energies, decay photon line
  intensities and energies, two-mode decay branching and a supplied flux
  spectrum's stated error. Other inputs (the self-shielding correction, the
  material composition) are held at nominal, and `get_data_uncertainty_info`
  lists every one it held. There are no first-order sensitivity coefficients.
- The decay data do not state how a nuclide's photon line intensities are
  correlated (between lines, between a spectrum's normalisation and its
  lines, between its gamma and x-ray spectra), and ENDF/B-VIII.1 folds each
  spectrum's normalisation sigma into every line's. So a photon spectrum,
  contact dose or decay heat is reported as a range rather than assumed:
  `Estimate.std_dev` takes those correlations as zero and
  `Estimate.std_dev_correlated` as one, which bounds every non-negative
  correlation. `get_data_uncertainty_info()["decay_photon_spectra_folded"]`
  names the spectra the range mostly comes from.
- Replicas are added 64 at a time until every tracked sigma (each nuclide
  density within 1e-6 of the largest, and the total activity, decay heat and
  decay photon line rate, at every step) has a standard error under 5% of
  itself, set with `DataUncertainty(convergence=...)`. The standard error
  carries the sample kurtosis, so a heavy-tailed output needs more replicas
  than a Gaussian one (about 201 at 5%). At least 128 and at most 1024 run;
  a run that stops on the cap says so (`hit_cap`) and lists the outputs that
  missed (`unconverged`).
- Damage energy and dpa carry no uncertainty yet: MT=444 has no covariance,
  and the flux and composition replicas are not folded into it.
- Pathways are reported per product (`get_production_routes`), but there is no
  automatic pathway search across the whole inventory.
- No ingestion or inhalation dose.
- Cross sections are read from the continuous-energy library and collapsed
  against your spectrum, so this reads transport-format data files even though
  it runs no transport. Only the sections activation needs are read, which is
  a small fraction of a library.
