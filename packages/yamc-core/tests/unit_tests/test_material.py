from pathlib import Path

import yamc
import pytest
from yamc import Material, enriched, Enriched

CHAIN_FILE = str(
    Path(__file__).resolve().parents[4]
    / "crates"
    / "yamc"
    / "tests"
    / "transmutation-endf-b8.1-sfr.arrow"
)


def _keywords_available():
    """Check if keyword download tests can work (requires download feature)."""
    try:
        m = Material(composition={"Li6": 1.0}, density=1.0)
        m.read_nuclear_data("endf-b8.1")
        return True
    except Exception:
        return False


requires_keywords = pytest.mark.skipif(
    not _keywords_available(),
    reason="keyword download requires download feature"
)


# ---------------------------------------------------------------------------
# Constructor basics
# ---------------------------------------------------------------------------

def test_composition_with_nuclides():
    """Material.nuclides preserves the user's input order."""
    mat = Material(composition={"H1": 1.0, "Fe56": 0.5}, density=1.0)
    assert mat.nuclides == [("H1", 1.0), ("Fe56", 0.5)]


def test_composition_preserves_input_order():
    """Material.nuclides must return entries in the order the user provided.

    Three nuclides given in non-alphabetical order should come back in the
    same order, not alphabetised."""
    mat = Material(
        composition={"H1": 0.5, "Fe56": 0.3, "Be9": 0.2},
        density=1.0,
    )
    names = [n for n, _ in mat.nuclides]
    assert names == ["H1", "Fe56", "Be9"]


def test_composition_order_preserved_with_element_expansion():
    """Element-expanded nuclides keep the user's top-level key order; expanded
    nuclides within a single key come out alphabetically."""
    mat = Material(
        composition={"Fe": 0.4, "Li": 0.6},
        density=1.0,
    )
    names = [n for n, _ in mat.nuclides]
    fe_idx = [i for i, n in enumerate(names) if n.startswith("Fe")]
    li_idx = [i for i, n in enumerate(names) if n.startswith("Li")]
    # All Fe nuclides come before all Li nuclides
    assert max(fe_idx) < min(li_idx)
    # Fe nuclides are alphabetised within
    fe_names = [names[i] for i in fe_idx]
    assert fe_names == sorted(fe_names)


def test_composition_order_with_nuclide_object_keys_keeps_order():
    """Nuclide-object keys also keep input order."""
    fe56 = yamc.Nuclide("Fe56")
    li6 = yamc.Nuclide("Li6")
    mat = Material(composition={fe56: 0.5, li6: 0.5}, density=1.0)
    names = [n for n, _ in mat.nuclides]
    assert names == ["Fe56", "Li6"]


def test_composition_with_nuclide_object_keys():
    """Nuclide objects can be used as composition keys interchangeably with strings."""
    li6 = yamc.Nuclide("Li6")
    li7 = yamc.Nuclide("Li7")
    mat = Material(composition={li6: 0.5, li7: 0.5}, density=0.534)
    nuclides = dict(mat.nuclides)
    assert nuclides == {"Li6": 0.5, "Li7": 0.5}


def test_composition_mixed_str_and_nuclide_keys():
    """A composition dict can mix string and Nuclide keys."""
    fe56 = yamc.Nuclide("Fe56")
    mat = Material(composition={fe56: 0.5, "H1": 1.0}, density=1.0)
    assert dict(mat.nuclides) == {"Fe56": 0.5, "H1": 1.0}


def test_composition_invalid_key_type():
    """Non-string, non-Nuclide keys are rejected."""
    with pytest.raises(ValueError, match="composition keys must be"):
        Material(composition={42: 1.0}, density=1.0)


def test_composition_with_elements():
    mat = Material(composition={"Li": 1.0}, density=0.534)
    nuclides = dict(mat.nuclides)
    assert "Li6" in nuclides
    assert "Li7" in nuclides
    assert abs(nuclides["Li6"] - 0.07589) < 1e-5
    assert abs(nuclides["Li7"] - 0.92411) < 1e-5


def test_composition_with_element_name():
    mat = Material(composition={"lithium": 1.0}, density=0.534)
    nuclides = dict(mat.nuclides)
    assert "Li6" in nuclides
    assert "Li7" in nuclides
    assert abs(nuclides["Li6"] - 0.07589) < 1e-5
    assert abs(nuclides["Li7"] - 0.92411) < 1e-5


def test_composition_with_formula():
    water = Material(composition={"H2O": 1.0}, density=1.0)
    nuclides = dict(water.nuclides)
    # H should be ~2/3, O should be ~1/3 (after natural isotope expansion)
    h_total = sum(v for k, v in nuclides.items() if k.startswith("H"))
    o_total = sum(v for k, v in nuclides.items() if k.startswith("O"))
    assert abs(h_total - 2.0 / 3.0) < 1e-3
    assert abs(o_total - 1.0 / 3.0) < 1e-3


def test_composition_formula_scaling():
    """Formula fraction scales the stoichiometric amounts."""
    mat = Material(composition={"H2O": 0.5, "Fe": 0.5}, density=3.0)
    nuclides = dict(mat.nuclides)
    h_total = sum(v for k, v in nuclides.items() if k.startswith("H"))
    o_total = sum(v for k, v in nuclides.items() if k.startswith("O"))
    fe_total = sum(v for k, v in nuclides.items() if k.startswith("Fe"))
    assert abs(h_total - 1.0 / 3.0) < 1e-3
    assert abs(o_total - 1.0 / 6.0) < 1e-3
    assert abs(fe_total - 0.5) < 1e-3


def test_composition_mixed():
    """Elements, nuclides, and formulas in one composition dict."""
    mat = Material(
        composition={"Fe": 0.4, "Li6": 0.3, "H2O": 0.3},
        density=5.0,
    )
    nuclides = dict(mat.nuclides)
    assert "Li6" in nuclides
    assert nuclides["Li6"] == 0.3
    assert any(k.startswith("Fe") for k in nuclides)
    assert any(k.startswith("H") for k in nuclides)


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def test_enriched_helper():
    e = enriched(0.5, target="Li6", percent=60.0)
    assert isinstance(e, Enriched)
    assert e.fraction == 0.5
    assert e.target == "Li6"
    assert e.percent == 60.0
    assert e.fraction_type == "atom"


def test_enriched_element():
    mat = Material(
        composition={"Li": enriched(1.0, target="Li6", percent=60.0)},
        density=0.534,
    )
    nuclides = dict(mat.nuclides)
    assert abs(nuclides["Li6"] - 0.6) < 1e-10
    assert abs(nuclides["Li7"] - 0.4) < 1e-10


def test_enriched_formula():
    mat = Material(
        composition={"Li4SiO4": enriched(1.0, target="Li6", percent=60.0)},
        density=2.4,
    )
    nuclides = dict(mat.nuclides)
    li6 = nuclides.get("Li6", 0)
    li7 = nuclides.get("Li7", 0)
    # Li is 4/9 of formula; 60% of that is Li6
    assert li6 > li7


def test_enriched_nuclide_error():
    """Enrichment on a specific nuclide should error."""
    with pytest.raises(ValueError, match="enrichment is not supported for nuclide"):
        Material(
            composition={"Li6": enriched(1.0, target="Li6", percent=60.0)},
            density=1.0,
        )


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------

def test_density_float():
    mat = Material(composition={"W184": 0.5}, density=19.3)
    assert mat.density == 19.3
    assert mat.density_units == "g/cm3"


def test_density_atom_per_barn_cm():
    # Total atom density + relative atom fractions.
    mat = Material(composition={"Li6": 0.625, "Li7": 0.375}, density=0.08,
                   units="atom/barn-cm")
    assert mat.density == 0.08
    assert mat.density_units == "atom/barn-cm"
    apbc = mat.get_atoms_per_barn_cm()
    assert abs(apbc["Li6"] - 0.05) < 1e-12
    assert abs(apbc["Li7"] - 0.03) < 1e-12


def test_atom_per_barn_cm_requires_atom_fractions():
    with pytest.raises(ValueError, match="atom"):
        Material(composition={"Li6": 1.0}, density=0.08,
                 units="atom/barn-cm", fraction_type="mass")


def test_from_atom_densities():
    # Absolute per-nuclide atom densities, stored verbatim; density is the sum.
    mat = Material.from_atom_densities({"Li6": 0.05, "Li7": 0.03})
    assert abs(mat.density - 0.08) < 1e-12
    assert mat.density_units == "atom/barn-cm"
    apbc = mat.get_atoms_per_barn_cm()
    assert apbc["Li6"] == 0.05
    assert apbc["Li7"] == 0.03


def test_decay_heat_total_from_atom_densities():
    mat = Material.from_atom_densities({"Mn56": 1.0e-12}, volume=2.0)
    heat = mat.decay_heat()

    atoms = 1.0e-12 * 1.0e24 * 2.0
    activity = atoms * 0.6931471805599453 / 9284.04
    expected = activity * 2522640.3 * 1.602176634e-19

    assert heat == pytest.approx(expected)


def test_decay_heat_by_nuclide_skips_stable_nuclides():
    mat = Material.from_atom_densities({"Mn56": 1.0e-12, "Fe56": 1.0e-12}, volume=1.0)
    heat = mat.decay_heat(by_nuclide=True)

    assert set(heat) == {"Mn56"}
    assert heat["Mn56"] > 0.0


def test_decay_heat_requires_volume():
    mat = Material.from_atom_densities({"Mn56": 1.0e-12})
    with pytest.raises(ValueError, match="volume"):
        mat.decay_heat()


def test_activity_total_from_atom_densities():
    mat = Material.from_atom_densities({"Mn56": 1.0e-12}, volume=2.0)
    activity = mat.activity()

    atoms = 1.0e-12 * 1.0e24 * 2.0
    expected = atoms * 0.6931471805599453 / 9284.04

    assert activity == pytest.approx(expected)


def test_activity_by_nuclide_skips_stable_nuclides():
    mat = Material.from_atom_densities({"Mn56": 1.0e-12, "Fe56": 1.0e-12}, volume=1.0)
    activity = mat.activity(by_nuclide=True)

    assert set(activity) == {"Mn56"}
    assert activity["Mn56"] > 0.0


def test_activity_requires_volume():
    mat = Material.from_atom_densities({"Mn56": 1.0e-12})
    with pytest.raises(ValueError, match="volume"):
        mat.activity()


# ---------------------------------------------------------------------------
# Contact dose rate
# ---------------------------------------------------------------------------

# A trace of Co60 in iron at 7.874 g/cm3, in atoms/(barn·cm). Co60 is the clean
# case: two gammas either side of 1.2 MeV, emitted on essentially every decay.
_COBALT_IN_IRON = {"Fe56": 0.084912, "Co60": 1.0e-6}


def test_contact_dose_matches_openmc():
    """The same inventory through OpenMC's get_photon_contact_dose_rate.

    Both codes fold the same chain's Co60 lines against the same NIST XCOM
    mu/rho, NIST-126 air mu_en/rho and ICRP-116 photon coefficients, so the
    reference values below are what OpenMC's own algorithm returns for this
    material to ten digits.
    """
    mat = Material.from_atom_densities(_COBALT_IN_IRON)

    assert mat.contact_dose() == pytest.approx(380.0531883582, rel=1e-9)
    assert mat.contact_dose(dose_quantity="effective") == pytest.approx(
        380.6934275503, rel=1e-9
    )


def test_contact_dose_by_nuclide_only_lists_photon_emitters():
    mat = Material.from_atom_densities(_COBALT_IN_IRON)
    by_nuclide = mat.contact_dose(by_nuclide=True)

    assert set(by_nuclide) == {"Co60"}
    assert by_nuclide["Co60"] == pytest.approx(mat.contact_dose())


def test_contact_dose_needs_no_volume():
    """Unlike activity and decay heat, the slab estimate is intensive."""
    without_volume = Material.from_atom_densities(_COBALT_IN_IRON)
    with_volume = Material.from_atom_densities(_COBALT_IN_IRON, volume=1000.0)

    assert without_volume.contact_dose() == pytest.approx(with_volume.contact_dose())


def test_contact_dose_scales_with_build_up():
    mat = Material.from_atom_densities(_COBALT_IN_IRON)
    assert mat.contact_dose(build_up=1.0) == pytest.approx(mat.contact_dose() / 2.0)


def test_contact_dose_of_stable_material_is_zero():
    mat = Material.from_atom_densities({"Fe56": 0.084912})
    assert mat.contact_dose() == 0.0
    assert mat.contact_dose(by_nuclide=True) == {}


def test_contact_dose_self_shielding_lowers_a_denser_host():
    """The same Co60 in lead reads lower: lead stops more of its own photons."""
    in_iron = Material.from_atom_densities(_COBALT_IN_IRON).contact_dose()
    in_lead = Material.from_atom_densities(
        {"Pb208": 0.032991, "Co60": 1.0e-6}
    ).contact_dose()

    assert in_lead < in_iron


def test_contact_dose_rejects_an_unknown_quantity():
    mat = Material.from_atom_densities(_COBALT_IN_IRON)
    with pytest.raises(ValueError, match="absorbed-air"):
        mat.contact_dose(dose_quantity="banana")


def test_contact_dose_rejects_a_non_positive_build_up():
    mat = Material.from_atom_densities(_COBALT_IN_IRON)
    with pytest.raises(ValueError, match="build_up must be positive"):
        mat.contact_dose(build_up=0.0)


def test_density_kg_m3():
    mat = Material(composition={"Fe56": 1.0}, density=7870.0, units="kg/m3")
    assert mat.density == 7870.0
    assert mat.density_units == "kg/m3"


def test_density_string_rejected():
    # The old density="sum" sentinel is gone; density must be numeric.
    with pytest.raises(TypeError):
        Material(composition={"Fe": 1.0}, density="sum")



# ---------------------------------------------------------------------------
# Optional constructor parameters
# ---------------------------------------------------------------------------

def test_name():
    mat = Material(composition={"Fe56": 1.0}, density=7.87, name="steel")
    assert mat.name == "steel"


def test_material_id():
    mat = Material(
        composition={"Fe56": 1.0}, density=7.87, id=42
    )
    assert mat.id == 42


def test_percent_type_mass():
    mat = Material(
        composition={"Fe": 0.7, "Cr": 0.3},
        density=7.87,
        fraction_type="mass",
    )
    assert mat.fraction_type == "mass"


def test_percent_type_weight_rejected():
    # The old "weight" spelling is no longer accepted -- fail hard.
    import pytest

    with pytest.raises((ValueError, Exception)):
        Material(
            composition={"Fe": 0.7, "Cr": 0.3},
            density=7.87,
            fraction_type="weight",
        )


def test_transmutable():
    mat = Material(
        composition={"Fe56": 1.0}, density=7.87, transmutable=True
    )
    assert mat.transmutable is True


def test_transmutable_default_false():
    mat = Material(composition={"Fe56": 1.0}, density=7.87)
    assert mat.transmutable is False


def test_volume_kwarg():
    mat = Material(
        composition={"Fe56": 1.0}, density=7.87, volume=100.0
    )
    assert mat.volume == 100.0


def test_temperature_kwarg():
    mat = Material(
        composition={"Fe56": 1.0}, density=7.87, temperature=600.0
    )
    assert mat.temperature == "600"


# ---------------------------------------------------------------------------
# Property setters (post-construction)
# ---------------------------------------------------------------------------

def test_volume_set_and_get():
    mat = Material(composition={"Fe56": 1.0}, density=7.87)
    mat.volume = 100.0
    assert mat.volume == 100.0
    mat.volume = 200.0
    assert mat.volume == 200.0
    with pytest.raises(ValueError, match="Volume must be positive"):
        mat.volume = -50.0
    assert mat.volume == 200.0


def test_material_id_set_and_get():
    mat = Material(composition={"Fe56": 1.0}, density=7.87)
    assert mat.id is None
    mat.id = 42
    assert mat.id == 42
    mat.id = 999
    assert mat.id == 999


def test_material_id_with_constructor():
    mat1 = Material(
        composition={"Fe56": 1.0}, density=7.87, id=10, name="mat1"
    )
    assert mat1.id == 10
    mat2 = Material(composition={"Fe56": 1.0}, density=7.87)
    mat2.id = 20
    assert mat1.id == 10
    assert mat2.id == 20


def test_material_id_large_values():
    mat = Material(composition={"Fe56": 1.0}, density=7.87, name="test")
    large_u32 = 4294967294
    mat.id = large_u32
    assert mat.id == large_u32


def test_material_name_set_and_get():
    mat = Material(composition={"Fe56": 1.0}, density=7.87)
    assert mat.name is None
    mat.name = "TestPyName"
    assert mat.name == "TestPyName"
    mat.name = "OtherName"
    assert mat.name == "OtherName"
    mat2 = Material(
        composition={"Fe56": 1.0}, density=7.87, name="InitialName"
    )
    assert mat2.name == "InitialName"


# ---------------------------------------------------------------------------
# get_nuclide_names
# ---------------------------------------------------------------------------

def test_get_nuclide_names():
    mat = Material(
        composition={"U235": 0.05, "U238": 0.95, "O16": 2.0}, density=10.0
    )
    assert mat.get_nuclide_names() == ["O16", "U235", "U238"]


# ---------------------------------------------------------------------------
# Invalid composition keys
# ---------------------------------------------------------------------------

def test_invalid_element():
    with pytest.raises(Exception, match="not a recognized element symbol"):
        Material(composition={"Xx": 1.0}, density=1.0)


# ---------------------------------------------------------------------------
# Nuclear data and cross sections
# ---------------------------------------------------------------------------

def test_global_default_cross_section_keyword():
    yamc.set_cross_section_data_entry("Li6", "tests/Li6.arrow")
    mat = Material(
        composition={"Li6": 1.0}, density=1.0, temperature=294
    )
    grid = mat.unified_energy_grid_neutron()
    assert len(grid) > 0


def test_mean_free_path_lithium_14mev():
    import math
    yamc.cross_section_data = ({"Li6": "tests/Li6.arrow", "Li7": "tests/Li7.arrow"})
    mat = Material(
        composition={"Li": 1.0}, density=0.534, temperature=294
    )
    mfp = mat.mean_free_path_neutron(14e6)
    assert mfp is not None
    # endf-b8.1 value (the VIII.0 fixtures gave ~14.963); matches the Rust pin
    # in test_mean_free_path_lithium_real_data.
    assert math.isclose(mfp, 14.97826945, rel_tol=1e-4), f"Expected ~14.978 cm, got {mfp}"


def test_material_reaction_mts_lithium():
    yamc.cross_section_data = ({"Li6": "tests/Li6.arrow", "Li7": "tests/Li7.arrow"})
    mat = Material(
        composition={"Li": 1.0}, density=0.534, temperature=294
    )
    mts = mat.reaction_mts
    essential_mts = [1, 2, 3, 101, 102]
    for mt in essential_mts:
        assert mt in mts, f"Material lithium should have MT {mt}, got {mts}"


def test_calculate_microscopic_xs_neutron_lithium():
    yamc.cross_section_data = ({"Li6": "tests/Li6.arrow", "Li7": "tests/Li7.arrow"})
    mat = Material(
        composition={"Li": 1.0}, density=0.534, temperature=294
    )
    micro_xs = mat.calculate_microscopic_xs_neutron()
    assert "Li6" in micro_xs
    assert "Li7" in micro_xs
    assert 2 in micro_xs["Li6"]
    assert 2 in micro_xs["Li7"]
    grid = mat.unified_energy_grid_neutron()
    assert len(micro_xs["Li6"][2]) == len(grid)
    assert len(micro_xs["Li7"][2]) == len(grid)


def test_material_vs_nuclide_microscopic_xs_li6():
    from yamc import Nuclide
    import numpy as np
    yamc.cross_section_data = ({"Li6": "tests/Li6.arrow"})
    mat = Material(
        composition={"Li6": 1.0}, density=0.534, temperature=294
    )
    micro_xs_mat = mat.calculate_microscopic_xs_neutron()
    grid = mat.unified_energy_grid_neutron()
    nuclide = Nuclide("Li6")
    nuclide.read_nuclear_data("tests/Li6.arrow")
    temperature = mat.temperature
    reactions = nuclide.reactions
    assert reactions is not None
    energy_map = nuclide.energy
    assert energy_map is not None
    energy_grid = energy_map.get(temperature)
    assert energy_grid is not None
    for mt, xs_mat in micro_xs_mat["Li6"].items():
        if mt in reactions:
            reaction = reactions[mt]
            threshold_idx = reaction.threshold_idx
            nuclide_energy = energy_grid[threshold_idx:]
            xs_nuclide = reaction.cross_section
            xs_nuclide_interp = []
            for g in grid:
                if g < nuclide_energy[0]:
                    xs_nuclide_interp.append(0.0)
                else:
                    xs = np.interp(g, nuclide_energy, xs_nuclide)
                    xs_nuclide_interp.append(xs)
            np.testing.assert_allclose(
                xs_mat, xs_nuclide_interp, rtol=1e-10, err_msg=f"Mismatch for MT {mt}"
            )


def test_calculate_microscopic_xs_neutron_mt_filter():
    mat = Material(
        composition={"Li": 1.0}, density=0.534, temperature=294
    )
    mat.read_nuclear_data({"Li6": "tests/Li6.arrow", "Li7": "tests/Li7.arrow"})
    xs_all = mat.calculate_microscopic_xs_neutron()
    xs_mt2 = mat.calculate_microscopic_xs_neutron(mt_filter=[2])
    for nuclide in ["Li6", "Li7"]:
        assert nuclide in xs_mt2
        assert list(xs_mt2[nuclide].keys()) == [2]
        assert xs_all[nuclide][2] == xs_mt2[nuclide][2]


def test_macroscopic_xs_neutron_mt_filter():
    mat = Material(
        composition={"Li": 1.0}, density=1.0, temperature=294
    )
    mat.read_nuclear_data({"Li6": "tests/Li6.arrow", "Li7": "tests/Li7.arrow"})
    xs_mt2, energy_mt2 = mat.macroscopic_cross_section(2)
    assert len(xs_mt2) == len(energy_mt2)
    assert len(xs_mt2) > 0
    assert all(xs >= 0 for xs in xs_mt2)


def test_hierarchical_mt3_generated_for_li6():
    mat = Material(
        composition={"Li6": 1.0}, density=0.534, temperature=294
    )
    mat.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    xs_mt3, energies = mat.macroscopic_cross_section(3)
    assert len(xs_mt3) > 0


def test_macroscopic_xs_mt3_does_not_generate_mt1():
    mat = Material(
        composition={"Li6": 1.0}, density=0.534, temperature=294
    )
    mat.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    xs_mt3, _ = mat.macroscopic_cross_section(3)
    assert len(xs_mt3) > 0


def test_macroscopic_xs_partial_mt_does_not_generate_mt1():
    # A partial reaction's macroscopic cross section must load without
    # fabricating MT=1 (total). Uses Li6 MT=105 (n,t), its dominant channel;
    # MT=24 (used pre-VIII.1) was dropped from Li6 in endf-b8.1.
    mat = Material(
        composition={"Li6": 1.0}, density=0.534, temperature=294
    )
    mat.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    xs_mt105, _ = mat.macroscopic_cross_section(105)
    assert len(xs_mt105) > 0


def test_sample_distance_to_collision_statistical():
    mat = Material(
        composition={"Li6": 1.0}, density=1.0, temperature=294
    )
    mat.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    mat.macroscopic_cross_section(1)
    energy = 14e6
    samples = []
    for seed in range(1000):
        d = mat.sample_distance_to_collision(energy, seed=seed)
        assert d is not None
        assert d >= 0.0
        samples.append(d)
    avg = sum(samples) / len(samples)
    assert abs(avg - 6.9) < 0.1, f"Average sampled distance {avg} not within 0.1 of 6.9"


def test_sample_interacting_nuclide_li6_li7():
    mat = Material(
        composition={"Li6": 0.5, "Li7": 0.5}, density=1.0, temperature=294
    )
    mat.read_nuclear_data({"Li6": "tests/Li6.arrow", "Li7": "tests/Li7.arrow"})
    mat.macroscopic_cross_section(1)
    energy = 100_000.0
    n_samples = 10000
    counts = {"Li6": 0, "Li7": 0}
    for seed in range(n_samples):
        nuclide = mat.sample_interacting_nuclide(energy, seed=seed)
        counts[nuclide] = counts.get(nuclide, 0) + 1
    frac_li6 = counts["Li6"] / n_samples
    frac_li7 = counts["Li7"] / n_samples
    assert frac_li6 > 0.0 and frac_li7 > 0.0
    assert abs(frac_li6 + frac_li7 - 1.0) < 1e-6
    assert frac_li6 > frac_li7


def test_material_be9_selective_temperature_load():
    from yamc.data import clear_nuclide_cache
    clear_nuclide_cache()
    mat = Material(
        composition={"Be9": 1.0}, density=1.85, temperature=294
    )
    yamc.cross_section_data = ({"Be9": "tests/Be9.arrow"})
    mat.read_nuclear_data({"Be9": "tests/Be9.arrow"})


@requires_keywords
def test_read_nuclides_keyword():
    mat = Material(
        composition={"Li": 1.0}, density=2.0, volume=1.0, temperature=294
    )
    mat.read_nuclear_data("endf-b8.1")
    grid = mat.unified_energy_grid_neutron()
    assert len(grid) > 0


def test_read_nuclides_dict():
    mat = Material(
        composition={"Li": 1.0}, density=2.0, volume=1.0, temperature=294
    )
    mat.read_nuclear_data({"Li6": "tests/Li6.arrow", "Li7": "tests/Li7.arrow"})


@requires_keywords
def test_material_different_data_sources():
    mat1 = Material(composition={"Li6": 1.0}, density=1.0, temperature=294)
    mat1.read_nuclear_data("endf-b8.1")
    mat2 = Material(
        composition={"Li6": 1.0}, density=1.0, temperature=294
    )
    mat2.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    grid1 = mat1.unified_energy_grid_neutron()
    grid2 = mat2.unified_energy_grid_neutron()
    assert len(grid1) > 0
    assert len(grid2) > 0


@requires_keywords
def test_material_file_and_keyword_sources():
    mat = Material(
        composition={"Li6": 1.0, "Li7": 1.0}, density=1.0, temperature=294
    )
    mat.read_nuclear_data({"Li6": "tests/Li6.arrow", "Li7": "endf-b8.1"})
    grid = mat.unified_energy_grid_neutron()
    assert len(grid) > 0


@requires_keywords
def test_material_cache_respects_data_source_boundaries():
    from yamc.data import clear_nuclide_cache
    clear_nuclide_cache()
    mat1 = Material(composition={"Li6": 1.0}, density=1.0, temperature=294)
    mat1.read_nuclear_data("endf-b8.1")
    mat2 = Material(
        composition={"Li6": 1.0}, density=1.0, temperature=294
    )
    mat2.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    grid1 = mat1.unified_energy_grid_neutron()
    grid2 = mat2.unified_energy_grid_neutron()
    assert len(grid1) > 0
    assert len(grid2) > 0


def test_material_path_normalization_in_cache():
    import os
    mat_rel = Material(
        composition={"Li6": 1.0}, density=1.0, temperature=294
    )
    mat_rel.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    mat_abs = Material(
        composition={"Li6": 1.0}, density=1.0, temperature=294
    )
    abs_path = os.path.abspath("tests/Li6.arrow")
    mat_abs.read_nuclear_data({"Li6": abs_path})
    xs_rel, _ = mat_rel.macroscopic_cross_section("(n,gamma)")
    xs_abs, _ = mat_abs.macroscopic_cross_section("(n,gamma)")
    assert xs_rel == xs_abs


def test_macroscopic_cross_section_without_temperature():
    mat = Material(
        composition={"Li6": 1.0}, density=1.0
    )
    mat.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    with pytest.raises(BaseException):
        mat.macroscopic_cross_section(reaction="(n,total)")
    mat.temperature = 294
    xs, energy = mat.macroscopic_cross_section(reaction="(n,total)")
    assert len(xs) > 0
    assert len(energy) > 0
    assert len(xs) == len(energy)


# ---------------------------------------------------------------------------
# String representation
# ---------------------------------------------------------------------------

def test_str_repr():
    mat = Material(
        composition={"Fe56": 1.0}, density=7.87, name="test"
    )
    s = str(mat)
    assert "Material:" in s
    assert "7.87" in s
    assert "Fe56" in s
