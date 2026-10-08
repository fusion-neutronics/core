"""Nuclear-data uncertainty on Material.transmute(), at the Python boundary.

The numerical work is pinned in Rust, where a fixture directory carrying
``covariance.arrow`` can be written on the fly (see
``crates/yani-transmute/tests/data_uncertainty.rs``). What is pinned HERE is the
contract a Python caller actually meets:

- asking for nothing gets you exactly what it always did, and ``None`` rather
  than a zero, so "not requested" is distinguishable from "no uncertainty";
- asking for it against data that carries no covariance gets you a report
  saying so, rather than a confident zero. That distinction is the whole point
  of the feature, and it is the one that is easy to lose at a binding.

The fixtures are fetched without ``covariance.arrow``, which is what makes them
the right data for the second case. They are links into (on Windows, copies of)
the download cache, though, which other tests fill with covariance now that the
published data carries it, so this module moves any ``covariance.arrow`` there
aside while it runs.
"""

import pathlib

import pytest
import yamc

NUC_DATA = "tests"
DAY = 86400.0

ENERGY_GROUPS = [1e-5, 0.625, 1e5, 2e7]
MULTIGROUP_FLUX = [1e12, 5e12, 1e14]
RATE = sum(MULTIGROUP_FLUX)


@pytest.fixture(autouse=True, scope="module")
def _fixtures_without_covariance():
    """Hide every fixture's ``covariance.arrow`` for this module, restoring it
    afterwards: the tests here are about data that carries none."""
    root = pathlib.Path(__file__).resolve().parents[4] / "tests"
    hidden = []
    for cov in root.glob("*.arrow/covariance.arrow"):
        aside = cov.with_name("covariance.arrow.hidden-by-test")
        cov.rename(aside)
        hidden.append((aside, cov))
    try:
        yield
    finally:
        for aside, cov in hidden:
            aside.rename(cov)


@pytest.fixture(autouse=True)
def _set_cross_sections():
    yamc.cross_section_data = NUC_DATA
    yield
    yamc.cross_section_data = None


def _iron():
    return yamc.Material(
        composition={"Fe56": 1.0},
        density=7.87,
        name="iron",
        volume=1.0,
        temperature=294,
    )


def _schedule():
    spectrum = yamc.NeutronSource(
        energy=yamc.sources.Histogram(ENERGY_GROUPS, MULTIGROUP_FLUX)
    )
    return yamc.PulseSchedule([
        yamc.Pulse(rate=RATE, duration=DAY, source=spectrum),
        yamc.Cooldown(duration=DAY),
    ])


# --- the option itself --------------------------------------------------------

def test_data_uncertainty_round_trips_its_arguments():
    u = yamc.DataUncertainty(seed=42)
    assert u.seed == 42
    assert u.samples is None, "adaptive by default: no user-facing sample count"

    fixed = yamc.DataUncertainty(seed=7, samples=128)
    assert (fixed.seed, fixed.samples) == (7, 128)


def test_zero_samples_is_rejected_rather_than_meaning_adaptive():
    # `samples=0` is the kind of thing that would otherwise be read as "use the
    # default", producing a silently unsampled run.
    with pytest.raises(ValueError, match="at least 1"):
        yamc.DataUncertainty(samples=0)


# --- not asked for ------------------------------------------------------------

def test_without_the_option_there_is_no_uncertainty_to_read():
    iron = _iron()
    results = iron.transmute(schedule=_schedule())
    mid = iron.id or 0

    assert results.get_nuclide_uncertainty(mid, "Mn56", 1) is None
    assert results.get_nuclide_uncertainty_evolution(mid, "Mn56") is None
    assert results.get_uncertainty_inventories(mid, 1) is None
    assert results.get_activity_uncertainty(mid, 1) is None
    assert results.get_decay_heat_uncertainty(mid, 1) is None
    assert results.get_decay_heat_uncertainty(mid, 1, by_nuclide=True) is None
    assert results.get_contact_dose_uncertainty(mid, 1) is None
    assert results.get_decay_photon_spectrum_uncertainty(mid, 1) is None
    assert results.get_data_uncertainty_info(iron.id or 0) is None


def test_the_default_signature_is_unchanged():
    """A positional-schedule call still works, so nothing existing breaks."""
    iron = _iron()
    results = iron.transmute(_schedule())
    assert results.num_steps == 2


# --- asked for, against data that has no covariance ---------------------------

def test_asking_on_data_without_covariance_reports_it_rather_than_a_zero():
    iron = _iron()
    results = iron.transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(seed=1, samples=8, sources=["cross_sections"]),
    )
    mid = iron.id or 0

    info = results.get_data_uncertainty_info(iron.id or 0)
    assert info is not None, "asking for uncertainty must produce a report"

    # The fixtures carry no covariance.arrow here (see the module fixture),
    # so nothing can be perturbed. The report has to say that.
    assert info["perturbed"] == []
    assert info["has_gaps"] is True
    assert "Fe56" in info["no_covariance_data"]

    # And the sigma reads as a number, not as None: uncertainty WAS requested.
    # It is zero because the evaluation says nothing, which is precisely why
    # `no_covariance_data` has to be checked alongside it.
    sigma = results.get_nuclide_uncertainty(mid, "Mn56", 1)
    assert sigma == 0.0


def test_skipped_block_counters_are_per_nuclide_dicts():
    """The skipped block counters are keyed by nuclide, not summed to one int.

    ``skipped_cross_material``, ``skipped_other_file`` and ``skipped_nc`` map
    a nuclide to its count, ``mirrored_disagree`` maps
    ``"Nuclide (n,a) (n,b)"`` to a relative difference, and
    ``derived_opposing_uncorrelated`` maps ``"Nuclide kind"`` to a list of
    ``[a, b]`` reaction pairs, and ``lumped_covariance_not_assignable`` maps
    ``"Nuclide MT852"`` to the lumped reaction's components. Their shape is
    checked on every run, and is empty when the fixture carries no
    covariance.arrow.
    """
    iron = _iron()
    results = iron.transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(seed=1, samples=8, sources=["cross_sections"]),
    )
    info = results.get_data_uncertainty_info(iron.id or 0)
    counters = ("skipped_cross_material", "skipped_other_file", "skipped_nc")
    maps = (
        "mirrored_disagree",
        "derived_opposing_uncorrelated",
        "lumped_covariance_not_assignable",
    )
    for key in (*counters, *maps):
        assert isinstance(info[key], dict), f"{key} is {type(info[key])}"
        assert all(isinstance(k, str) for k in info[key]), info[key]
    for key in counters:
        for nuclide, count in info[key].items():
            assert isinstance(count, int) and count > 0, (key, nuclide, count)
    for pair, difference in info["mirrored_disagree"].items():
        assert len(pair.split(" ")) == 3, pair
        assert difference > 0.0, (pair, difference)
    for channel, pairs in info["derived_opposing_uncorrelated"].items():
        assert len(channel.split(" ")) == 2, channel
        assert isinstance(pairs, list) and pairs, (channel, pairs)
        for pair in pairs:
            assert isinstance(pair, list) and len(pair) == 2, (channel, pair)
            assert all(isinstance(name, str) for name in pair), (channel, pair)
    for lump, components in info["lumped_covariance_not_assignable"].items():
        nuclide, mtl = lump.split(" ")
        assert nuclide[:1].isupper() and nuclide.isalnum(), lump
        assert 851 <= int(mtl.removeprefix("MT")) <= 870, lump
        assert isinstance(components, list) and len(components) > 1, (lump, components)
    if "Fe56" not in info["perturbed"]:
        assert info["skipped_cross_material"] == {}
        assert info["skipped_other_file"] == {}
        assert info["skipped_nc"] == {}
        assert info["mirrored_disagree"] == {}
        assert info["derived_opposing_uncorrelated"] == {}
        assert info["lumped_covariance_not_assignable"] == {}


def test_the_report_names_what_is_never_perturbed():
    """The sources this does not propagate are stated, not left to be inferred."""
    iron = _iron()
    results = iron.transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(seed=1, samples=8),
    )
    not_perturbed = results.get_data_uncertainty_info(iron.id or 0)["not_perturbed"]
    joined = " ".join(not_perturbed)
    for source in (
        "fission yield",
        "isomeric branching (MF=9/MF=10)",
        "covariance with another evaluation (MAT1 naming another material)",
        "resonance-parameter covariance (MF=32)",
        "decay photon line energy and intensity (MF=8 MT=457)",
        "photon attenuation coefficient (XCOM)",
        "air energy-absorption coefficient (NIST SRD 126)",
        "fluence-to-dose coefficient (ICRP-116)",
        "contact-dose build-up factor",
        "material composition",
        "material density",
        "natural isotopic abundance",
        "atomic mass (AME2020)",
    ):
        assert source in joined, f"{source!r} missing from {not_perturbed}"
    # A dilute spectrum run has no shielding and no tally to hold fixed.
    assert "self-shielding correction" not in not_perturbed
    assert "flux response to perturbed cross sections (one transport)" not in not_perturbed
    assert "tallied-rate statistics" not in not_perturbed
    # Every default source is on, so none is listed as switched off.
    assert "activation cross section (MF=33)" not in not_perturbed
    assert "decay branching ratio" not in not_perturbed


def test_a_shielded_run_reports_its_shielding_held_at_nominal():
    """The shielded flux shape is built once, so the report says it was held."""
    iron = _iron()
    results = iron.transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(seed=1, samples=8),
        self_shielding_chord=2.0,
    )
    info = results.get_data_uncertainty_info(iron.id or 0)
    assert "self-shielding correction" in info["not_perturbed"]


def test_a_draw_on_relative_cells_alone_floors_nothing():
    """Iron's channels read only lognormal cell multipliers with positive
    coefficients, so no drawn rate can go negative and none is floored.

    Holds whether or not the fixture carries covariance: without it nothing
    is drawn at all, and both counters are reported anyway.
    """
    iron = _iron()
    results = iron.transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(seed=1, samples=8),
    )
    info = results.get_data_uncertainty_info(iron.id or 0)
    assert info["rates_floored"] == 0
    assert "rates_sampled" in info


def test_coverage_is_a_share_and_a_dilute_run_has_no_partials_off_the_rate():
    """Every coverage entry is a share, and nothing on a dilute run is off one.

    On an unshielded collapse the fold's partial rates and the rate they are
    divided by are the same integral, so ``partials_above_rate`` and
    ``partials_below_rate`` must be empty; an entry there is a channel whose
    sigma is overstated or understated.
    """
    iron = _iron()
    results = iron.transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(seed=1, samples=8, sources=["cross_sections"]),
    )
    info = results.get_data_uncertainty_info(iron.id or 0)
    # The key and its conversion are checked on every run: without covariance
    # it is an empty dict, which is valid.
    assert isinstance(info["partials_above_rate"], dict)
    assert isinstance(info["partials_below_rate"], dict)
    # Without covariance the checks below pass on empty maps, so a fixture
    # without it is skipped, visibly, rather than passed unexamined. Whether
    # the Fe56 fixture carries covariance.arrow depends on the cache: the URL
    # cache fetches it on demand once uncertainty is asked for. The same
    # checks are pinned on a built fixture in
    # crates/yani-transmute/tests/data_uncertainty.rs.
    if "Fe56" not in info["perturbed"]:
        pytest.skip("the Fe56 fixture carries no covariance.arrow")
    assert info["rate_fraction_covered"], "the fold consumed no covariance"
    assert info["partials_above_rate"] == {}
    assert info["partials_below_rate"] == {}
    for channel, share in info["rate_fraction_covered"].items():
        assert 0.0 <= share <= 1.0, f"{channel} reads {share}"


def test_the_means_are_unchanged_by_asking_for_uncertainty():
    """Bit-identical, not merely close: the nominal pass is untouched."""
    iron = _iron()
    plain = iron.transmute(schedule=_schedule())
    with_unc = _iron().transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(seed=3, samples=8),
    )
    mid = iron.id or 0

    a = plain.get_material_nuclides(mid, 1)
    b = with_unc.get_material_nuclides(mid, 1)
    assert a.keys() == b.keys()
    for name in a:
        assert a[name] == b[name], f"{name} moved when uncertainty was switched on"


def test_the_starting_inventory_carries_no_uncertainty():
    iron = _iron()
    results = iron.transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(seed=1, samples=8),
    )
    # Step 0 is the initial composition: an input, not a result.
    assert results.get_nuclide_uncertainty(iron.id or 0, "Fe56", 0) == 0.0


def test_the_uncertainty_evolution_lines_up_with_the_density_evolution():
    iron = _iron()
    results = iron.transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(seed=1, samples=8),
    )
    mid = iron.id or 0
    means = results.get_nuclide_evolution(mid, "Fe56")
    sigmas = results.get_nuclide_uncertainty_evolution(mid, "Fe56")
    assert len(means) == len(sigmas), "the two must zip without an index shift"
    assert sigmas[0] == 0.0


# --- choosing which sources to perturb ----------------------------------------

def test_available_sources_lists_what_this_build_can_do():
    # The list grows as sources land, so this asserts the contract rather than
    # the contents: every name it returns is accepted, and a name it does not
    # return is refused. That is what a caller actually needs from it.
    available = yamc.DataUncertainty.available_sources()
    assert available, "a build with no perturbable source would be useless"
    assert yamc.DataUncertainty(sources=available).sources == available
    with pytest.raises(ValueError):
        yamc.DataUncertainty(sources=["not_a_source"])


def test_sources_defaults_to_everything_implemented():
    u = yamc.DataUncertainty(seed=1)
    assert u.sources == yamc.DataUncertainty.available_sources()


def test_naming_a_source_that_does_not_exist_yet_raises():
    """The property the progressive workflow depends on.

    Adding sources one at a time and watching the inventory sigma grow only
    means something if asking for a source that has not landed is an error. A
    silently ignored ``fission_yield`` would look exactly like a
    ``fission_yield`` that contributed nothing, which is the one confusion this
    whole feature exists to prevent.
    """
    with pytest.raises(ValueError, match="fission_yield"):
        yamc.DataUncertainty(sources=["fission_yield"])

    with pytest.raises(ValueError, match="cross_sections"):
        # The message must name what IS available, not just what is not.
        yamc.DataUncertainty(sources=["typo"])


def test_an_empty_source_list_is_rejected_rather_than_meaning_everything():
    # `sources=[]` reads as "none" to a caller and would otherwise mean "all",
    # which is the wrong way round to guess.
    with pytest.raises(ValueError, match="at least one"):
        yamc.DataUncertainty(sources=[])


def test_the_repr_shows_the_sources():
    r = repr(yamc.DataUncertainty(seed=5, samples=32, sources=["cross_sections"]))
    assert "seed=5" in r and "samples=32" in r and "cross_sections" in r


def test_the_report_says_which_sources_were_on():
    iron = _iron()
    results = iron.transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(
            seed=1, samples=8, sources=["cross_sections"]
        ),
    )
    assert results.get_data_uncertainty_info(iron.id or 0)["sources"] == ["cross_sections"]


# --- flux spectrum uncertainty ------------------------------------------------

def _pulse(flux_std_dev=None):
    spectrum = yamc.NeutronSource(
        energy=yamc.sources.Histogram(ENERGY_GROUPS, MULTIGROUP_FLUX)
    )
    return yamc.Pulse(
        rate=RATE, duration=DAY, source=spectrum, flux_std_dev=flux_std_dev
    )


def test_flux_spectrum_is_an_available_source():
    assert "flux_spectrum" in yamc.DataUncertainty.available_sources()


def test_a_pulse_carries_an_optional_flux_error():
    # The common case: a spectrum from a published reference set has no stated
    # error, so the argument is omitted entirely.
    assert _pulse().flux_std_dev is None

    # From a yamc run it is results[tally].standard_deviation.
    sigma = [1e11, 5e11, 1e13]
    assert _pulse(sigma).flux_std_dev == sigma


def test_a_flux_error_needs_a_source_to_be_the_error_of():
    with pytest.raises(ValueError, match="needs a source"):
        yamc.Pulse(rate=RATE, duration=DAY, flux_std_dev=[1.0, 2.0, 3.0])


def test_a_negative_flux_error_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        _pulse([1e11, -5e11, 1e13])


def test_a_flux_error_of_the_wrong_length_is_refused():
    """A sigma that does not line up with the flux is not a sigma for it."""
    iron = _iron()
    schedule = yamc.PulseSchedule([_pulse([1e11, 5e11])])  # 2 entries, 3 bins
    with pytest.raises(ValueError, match="line up"):
        iron.transmute(
            schedule=schedule,
            data_uncertainty=yamc.DataUncertainty(seed=1, samples=4),
        )


def test_a_flux_error_moves_the_inventory_without_any_covariance():
    """Flux uncertainty needs no nuclear data at all -- it is the user's input.

    The committed fixtures carry no covariance, so this isolates the flux
    contribution cleanly: any spread at all must have come from the flux.
    """
    iron = _iron()
    mid = iron.id or 0
    # 10% on every bin.
    sigma = [0.10 * f for f in MULTIGROUP_FLUX]
    results = iron.transmute(
        schedule=yamc.PulseSchedule([_pulse(sigma), yamc.Cooldown(duration=DAY)]),
        data_uncertainty=yamc.DataUncertainty(
            seed=3, samples=128, sources=["flux_spectrum"]
        ),
    )

    mean = results.get_nuclide_density(mid, "Mn56", 1)
    spread = results.get_nuclide_uncertainty(mid, "Mn56", 1)
    assert mean > 0.0
    assert spread > 0.0, "a 10% flux error must move Mn56"

    info = results.get_data_uncertainty_info(iron.id or 0)
    assert info["spectra_with_flux_sigma"] == 1
    assert info["spectra_without_flux_sigma"] == 0


def test_a_spectrum_without_an_error_is_reported_not_assumed_exact():
    """The FISPACT reference-spectra case.

    A user with a published spectrum has no error for it. That has to stay
    distinguishable from a flux known to be exact.
    """
    iron = _iron()
    results = iron.transmute(
        schedule=yamc.PulseSchedule([_pulse(), yamc.Cooldown(duration=DAY)]),
        data_uncertainty=yamc.DataUncertainty(
            seed=1, samples=8, sources=["flux_spectrum"]
        ),
    )
    info = results.get_data_uncertainty_info(iron.id or 0)
    assert info["spectra_without_flux_sigma"] == 1
    assert info["spectra_with_flux_sigma"] == 0
    assert info["has_gaps"] is True
    assert results.get_nuclide_uncertainty(iron.id or 0, "Mn56", 1) == 0.0


# --- derived quantities -------------------------------------------------------
#
# Activity and decay heat are functions of a whole inventory, so their sigma has
# to come from evaluating the ensemble per replica. The two ways of getting a
# plausible wrong answer -- evaluating from the mean inventory, and adding the
# per-nuclide sigmas in quadrature -- are pinned against in Rust, where an
# ensemble that genuinely moves can be built (crates/yani-transmute/src/derived.rs).
# What is pinned HERE is the binding: that the accessor exists, that its nominal
# is the same number the existing API already gives, and that "not requested"
# stays distinguishable from "measured zero".


def _uncertain_results():
    # Cross sections only: the fixtures carry no covariance, so nothing is
    # sampled, which is the case these tests are about. The published decay
    # data does carry half-life sigmas, so the default sources would sample.
    iron = _iron()
    results = iron.transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(seed=1, samples=8, sources=["cross_sections"]),
    )
    return results, iron.id or 0


@pytest.mark.parametrize("quantity", ["activity", "decay_heat"])
def test_the_nominal_is_the_number_the_material_api_already_gives(quantity):
    """Bit-identical, not merely close.

    The accessor takes the volume off the same stored step material, so any gap
    here would mean the two are scaled differently -- the failure that would
    otherwise show up as a band drawn around the wrong centre.
    """
    results, mid = _uncertain_results()
    estimate = getattr(results, f"get_{quantity}_uncertainty")(mid, 1)
    on_the_material = getattr(results.get_material(mid, 1), quantity)()
    assert estimate.nominal == on_the_material


def test_no_covariance_gives_an_absent_spread_rather_than_a_confident_zero():
    """The state most runs against published data are actually in.

    The committed fixtures carry no covariance, so nothing can be resampled and
    zero replicas run. A spread over nothing is not zero, and `None` says so.

    This deliberately differs from ``get_nuclide_uncertainty``, which reports
    0.0 in the same situation and leans on
    ``get_data_uncertainty_info(id)["no_covariance_data"]`` to be checked
    alongside it.
    A zero-width band drawn around a decay heat is a claim of exactness that
    nobody would check; `None` cannot be plotted by accident.
    """
    results, mid = _uncertain_results()
    assert results.get_data_uncertainty_info(mid)["samples"] == 0

    estimate = results.get_activity_uncertainty(mid, 1)
    assert estimate.replicas == 0
    assert estimate.std_dev is None
    assert estimate.mean is None
    assert estimate.relative_std_dev is None
    assert estimate.std_dev_standard_error is None
    assert estimate.nominal > 0.0, "the unperturbed run still has an answer"

    # The neighbouring accessor's answer to the same question, for contrast.
    assert results.get_nuclide_uncertainty(mid, "Mn56", 1) == 0.0
    # Its standard error needs four replicas, and there are none.
    assert results.get_nuclide_uncertainty_standard_error(mid, "Mn56", 1) is None


def test_by_nuclide_agrees_with_the_material_breakdown_it_mirrors():
    results, mid = _uncertain_results()
    breakdown = results.get_activity_uncertainty(mid, 1, by_nuclide=True)
    on_the_material = results.get_material(mid, 1).activity(by_nuclide=True)

    assert on_the_material, "a fixture with nothing in it would prove nothing"
    assert set(breakdown) == set(on_the_material)
    for nuclide, becquerel in on_the_material.items():
        assert breakdown[nuclide].nominal == becquerel


def test_the_initial_composition_is_reported_like_any_other_step():
    """Step 0 is the initial composition, and reads as the input it is."""
    results, mid = _uncertain_results()
    estimate = results.get_activity_uncertainty(mid, 0)
    assert estimate.nominal == results.get_material(mid, 0).activity()


def test_the_estimate_repr_says_what_it_is():
    results, mid = _uncertain_results()
    text = repr(results.get_decay_heat_uncertainty(mid, 1))
    assert text.startswith("Estimate(nominal=")
    assert "replicas=0" in text


# --- the photon spectrum and the contact dose ---------------------------------
#
# The committed fixture chain carries no photon sources at all, so the spectrum
# is empty and the contact dose is zero against it. That makes these binding
# tests rather than numerical ones, which is the same split the rest of this
# file follows: the union-of-lines rule, the emitting count and the contact
# dose's cancellation under a global density scale are pinned in Rust, where a
# chain with Co60 in it can be written by hand
# (crates/yani-transmute/src/derived.rs).


def _no_volume_results():
    iron = yamc.Material(
        composition={"Fe56": 1.0}, density=7.87, name="iron", temperature=294
    )
    results = iron.transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(seed=1, samples=8),
    )
    return results, iron.id or 0


def test_the_contact_dose_is_the_one_derived_quantity_needing_no_volume():
    """It takes the material for a half-space, so a bigger lump reads the same.

    The other three count atoms and cannot answer without one. Having the same
    accessor shape refuse a volume for one quantity and require it for three is
    worth pinning, because it is the kind of thing a shared code path quietly
    makes uniform in the wrong direction.
    """
    results, mid = _no_volume_results()

    assert results.get_contact_dose_uncertainty(mid, 1) is not None

    for call in (
        lambda: results.get_activity_uncertainty(mid, 1),
        lambda: results.get_decay_heat_uncertainty(mid, 1),
        lambda: results.get_decay_photon_spectrum_uncertainty(mid, 1),
    ):
        with pytest.raises(ValueError, match="volume"):
            call()


def test_the_contact_dose_nominal_is_the_number_the_material_api_gives():
    results, mid = _uncertain_results()
    for dose_quantity in ("absorbed-air", "effective"):
        estimate = results.get_contact_dose_uncertainty(
            mid, 1, dose_quantity=dose_quantity, build_up=2.0
        )
        on_the_material = results.get_material(mid, 1).contact_dose(
            dose_quantity=dose_quantity, build_up=2.0
        )
        assert estimate.nominal == on_the_material


def test_an_unknown_dose_quantity_is_refused_as_it_is_on_the_material():
    results, mid = _uncertain_results()
    with pytest.raises(ValueError, match="absorbed-air"):
        results.get_contact_dose_uncertainty(mid, 1, dose_quantity="rem")


def test_the_photon_spectrum_uncertainty_mirrors_the_spectrum_it_annotates():
    """Same lines, same order, same nominal rates as the material's own."""
    results, mid = _uncertain_results()
    lines = results.get_decay_photon_spectrum_uncertainty(mid, 1)
    energies, rates = results.get_material(mid, 1).decay_photon_spectrum()

    assert [line.energy for line in lines] == energies
    assert [line.nominal for line in lines] == rates
    assert all(line.replicas == 0 for line in lines), "no covariance in the fixture"


def test_the_contact_dose_by_nuclide_is_a_dict_of_estimates():
    results, mid = _uncertain_results()
    breakdown = results.get_contact_dose_uncertainty(mid, 1, by_nuclide=True)
    on_the_material = results.get_material(mid, 1).contact_dose(by_nuclide=True)

    assert set(breakdown) == set(on_the_material)
    for nuclide, dose in on_the_material.items():
        assert breakdown[nuclide].nominal == dose


# --- half-life uncertainty ----------------------------------------------------
#
# The numerical rules (a saturated activity insensitive to its own half-life,
# the lambda t sensitivity after cooling) are pinned in Rust, in
# crates/yani-transmute/tests/half_life_uncertainty.rs. Here: the source is
# offered, and leaving it out is reported.

def test_half_life_is_an_available_source():
    assert "half_life" in yamc.DataUncertainty.available_sources()


def test_leaving_half_lives_out_is_reported():
    iron = _iron()
    results = iron.transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(
            seed=1, samples=8, sources=["cross_sections"]
        ),
    )
    info = results.get_data_uncertainty_info(iron.id or 0)
    assert "half-life" in info["not_perturbed"]
    assert info["half_lives_perturbed"] == []
    assert info["half_lives_sampled"] == 0


def test_asking_for_half_lives_reports_which_were_sampled():
    iron = _iron()
    results = iron.transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(seed=1, samples=8, sources=["half_life"]),
    )
    info = results.get_data_uncertainty_info(iron.id or 0)
    assert "half-life" not in info["not_perturbed"]
    # Every reachable unstable nuclide is in exactly one of the three lists.
    perturbed = set(info["half_lives_perturbed"])
    without = set(info["no_half_life_uncertainty"])
    not_carried = set(info["half_life_uncertainty_not_carried"])
    assert not perturbed & without
    assert not perturbed & not_carried
    assert not without & not_carried
    assert perturbed or without
    assert isinstance(info["half_life_uncertainty_not_carried"], list)
    assert isinstance(info["decay_energy_uncertainty_not_carried"], list)
    # The draw is lognormal, so there is nothing to floor and no count of it.
    assert "half_lives_floored" not in info



# --- flux covariance ----------------------------------------------------------
#
# The sampling is pinned in Rust (flux_uncertainty.rs and
# tests/flux_covariance.rs). Here: the argument, its checks, and that a
# correlated error reaches the inventory as one.

def _cov_pulse(correlation):
    """10% on every bin, with one correlation between every pair of bins."""
    sd = [0.10 * f for f in MULTIGROUP_FLUX]
    n = len(sd)
    cov = [
        [sd[i] * sd[j] * (1.0 if i == j else correlation) for j in range(n)]
        for i in range(n)
    ]
    spectrum = yamc.NeutronSource(
        energy=yamc.sources.Histogram(ENERGY_GROUPS, MULTIGROUP_FLUX)
    )
    return yamc.Pulse(rate=RATE, duration=DAY, source=spectrum, flux_covariance=cov)


def test_a_pulse_carries_an_optional_flux_covariance():
    pulse = _cov_pulse(0.5)
    assert pulse.flux_std_dev is None
    assert len(pulse.flux_covariance) == len(MULTIGROUP_FLUX)


def test_a_flux_error_is_one_or_the_other():
    spectrum = yamc.NeutronSource(
        energy=yamc.sources.Histogram(ENERGY_GROUPS, MULTIGROUP_FLUX)
    )
    with pytest.raises(ValueError, match="not both"):
        yamc.Pulse(
            rate=RATE,
            duration=DAY,
            source=spectrum,
            flux_std_dev=[1.0, 1.0, 1.0],
            flux_covariance=[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
        )


def test_a_covariance_that_is_not_one_is_refused_at_the_solve():
    iron = _iron()
    spectrum = yamc.NeutronSource(
        energy=yamc.sources.Histogram(ENERGY_GROUPS, MULTIGROUP_FLUX)
    )
    bad = yamc.Pulse(
        rate=RATE,
        duration=DAY,
        source=spectrum,
        flux_covariance=[[1.0, 2.0, 0], [2.0, 1.0, 0], [0, 0, 1.0]],
    )
    with pytest.raises(ValueError, match="positive semi-definite"):
        iron.transmute(schedule=yamc.PulseSchedule([bad]))
    wrong = yamc.Pulse(
        rate=RATE, duration=DAY, source=spectrum, flux_covariance=[[1.0, 0], [0, 1.0]]
    )
    with pytest.raises(ValueError, match="3 x 3"):
        iron.transmute(schedule=yamc.PulseSchedule([wrong]))


def test_correlated_bins_move_the_inventory_more_than_independent_ones():
    """Fe57 is made by capture, which the whole spectrum drives, so whether
    the bins move together decides how far the rate moves."""

    def spread(correlation):
        iron = _iron()
        results = iron.transmute(
            schedule=yamc.PulseSchedule([_cov_pulse(correlation)]),
            data_uncertainty=yamc.DataUncertainty(
                seed=3, samples=256, sources=["flux_spectrum"]
            ),
        )
        mid = iron.id or 0
        return results.get_nuclide_uncertainty(mid, "Fe57", 1) / results.get_nuclide_density(
            mid, "Fe57", 1
        )

    independent, correlated = spread(0.0), spread(1.0)
    assert correlated == pytest.approx(0.10, rel=0.1)
    assert independent < 0.9 * correlated


# --- attribution --------------------------------------------------------------
#
# The numbers are pinned in Rust (tests/uncertainty_attribution.rs, and the
# cross-section case in tests/data_uncertainty.rs). Here: the switch and the
# shape of what comes back.

def test_attribution_is_off_unless_asked():
    assert yamc.DataUncertainty().attribution is False
    iron = _iron()
    sigma = [0.10 * f for f in MULTIGROUP_FLUX]
    results = iron.transmute(
        schedule=yamc.PulseSchedule([_pulse(sigma)]),
        data_uncertainty=yamc.DataUncertainty(seed=1, samples=16, sources=["flux_spectrum"]),
    )
    assert results.get_uncertainty_breakdown(iron.id or 0, "Mn56", 1) is None


def test_the_breakdown_says_which_source_carries_the_variance():
    iron = _iron()
    sigma = [0.10 * f for f in MULTIGROUP_FLUX]
    results = iron.transmute(
        schedule=yamc.PulseSchedule([_pulse(sigma)]),
        data_uncertainty=yamc.DataUncertainty(
            seed=1, samples=64, sources=["flux_spectrum"], attribution=True
        ),
    )
    mid = iron.id or 0
    b = results.get_uncertainty_breakdown(mid, "Mn56", 1)
    assert b["variance"] == pytest.approx(results.get_nuclide_uncertainty(mid, "Mn56", 1) ** 2)
    # One source: it carries everything, and nothing is left over.
    assert b["by_source"] == {"flux_spectrum": b["variance"]}
    assert b["unattributed"] == 0.0
    assert isinstance(b["contributors"], list)
    # The flux has no first-order terms: its linearity is not applicable, not
    # read as nonlinear.
    assert b["linearity"] == {"flux_spectrum": None, "all": None}
    assert results.get_uncertainty_breakdown(mid, "Mn56", 0)["variance"] == 0.0


# --- decay energy -------------------------------------------------------------
#
# The component split and its uncertainty are pinned in Rust
# (yani-convert/tests/round_trip.rs, yani-decay, and
# yani-transmute/tests/decay_energy_uncertainty.rs). Here: the source is
# offered, and a component heat refuses data that carries no split.

def test_decay_energy_is_an_available_source():
    assert "decay_energy" in yamc.DataUncertainty.available_sources()


def test_a_component_is_named_or_refused():
    iron = _iron()
    with pytest.raises(ValueError, match="component must be one of"):
        iron.decay_heat(component="neutrino")


def test_with_no_rate_drawn_the_repair_report_is_empty():
    """With no rate drawn, the repair report is empty and exact."""
    iron = _iron()
    # Half-lives only, so no cross-section rate is drawn whatever covariance
    # the cached fixture carries, and the weighted inflation has nothing to
    # weigh. The values of a real repair and the nested layout are pinned in
    # Rust (covariance_sample.rs and the yani-python data_uncertainty.rs
    # tests).
    results = iron.transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(seed=1, samples=8, sources=["half_life"]),
    )
    info = results.get_data_uncertainty_info(iron.id or 0)
    assert info["rates_sampled"] == 0
    assert info["covariance_repaired"] == []
    assert info["covariance_repairs"] == []
    assert info["covariance_repaired_outside_bound"] == []
    assert info["worst_sigma_inflation"] == 0.0
    assert info["rate_weighted_sigma_inflation"] is None
    assert info["sigma_at_least_one"] == {}
    assert info["sigma_at_least_ten"] == {}
    assert info["sigma_at_least_one_outside_bound"] == {}
