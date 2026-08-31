#!/usr/bin/env python3
"""Tests for tally functionality."""

import pytest
import yamc


class TestTally:
    """Test tally creation and basic properties."""
    
    def test_tally_creation(self):
        """Test creating a new tally."""
        tally = yamc.Tally()
        assert tally.scores == []  # Default score
        assert tally.name is None
        assert tally.id is None
        assert tally.units == []  # No scores, so no units

    def test_tally_score_assignment(self):
        """Test setting tally score as single integer."""
        assert yamc.Tally(scores=[101]).scores == [101]
        assert yamc.Tally(scores=[2]).scores == [2]
        assert yamc.Tally(scores=[18]).scores == [18]

    def test_tally_score_type_validation(self):
        """Test that tally score only accepts integers or string scores."""
        with pytest.raises(TypeError):
            yamc.Tally(scores='101')  # Should be list, not string

        with pytest.raises(TypeError):
            yamc.Tally(scores=101)  # Should be list, not int

        with pytest.raises(TypeError):
            yamc.Tally(scores=101.0)  # Should be list, not float

    def test_tally_flux_score_string(self):
        """Test setting flux score using string."""
        assert yamc.Tally(scores=['flux']).scores == ['flux']
        assert yamc.Tally(scores=[101, 'flux']).scores == [101, 'flux']

        with pytest.raises(ValueError):
            yamc.Tally(scores=['invalid_score'])

    def test_tally_heating_scores(self):
        """Test setting heating scores using strings and MT numbers."""
        assert yamc.Tally(scores=['heating']).scores == ['heating']
        # MT 301 maps to 'heating'
        assert yamc.Tally(scores=['301']).scores == ['heating']
        assert yamc.Tally(scores=['heating-local']).scores == ['heating-local']
        # MT 901 maps to 'heating-local'
        assert yamc.Tally(scores=['901']).scores == ['heating-local']
        assert yamc.Tally(scores=['heating', 'heating-local']).scores == ['heating', 'heating-local']
        assert yamc.Tally(scores=['flux', 'heating', 101]).scores == ['flux', 'heating', 101]

    def test_tally_production_scores(self):
        """Test setting light particle production scores using strings."""
        for name in ['H1-production', 'H2-production', 'H3-production',
                     'He3-production', 'He4-production']:
            assert yamc.Tally(scores=[name]).scores == [name]

        # Test mixed with MT numbers and flux
        assert yamc.Tally(scores=['flux', 'H3-production', 101]).scores == \
            ['flux', 'H3-production', 101]

        # Test all production scores together
        all_prod = ['H1-production', 'H2-production', 'H3-production',
                    'He3-production', 'He4-production']
        assert yamc.Tally(scores=all_prod).scores == all_prod

    def test_tally_reaction_name_scores(self):
        """Test setting scores using ENDF reaction names like (n,2n), (n,gamma).

        Named variants preserve the user's input string:
        - 'total' -> 'total'
        - 'elastic' -> 'elastic'
        - 'fission' -> 'fission'
        - 'absorption' -> 'absorption'

        REACTION_MT lookups preserve the original string:
        - '(n,2n)' -> '(n,2n)'
        - '(n,gamma)' -> '(n,gamma)'
        """
        for name in ['total', 'elastic', 'fission', 'absorption',
                     '(n,2n)', '(n,gamma)', '(n,a)', '(n,p)']:
            assert yamc.Tally(scores=[name]).scores == [name]

        mixed = ['total', 'elastic', 'fission', '(n,2n)', '(n,gamma)']
        assert yamc.Tally(scores=mixed).scores == mixed

        with_basic = ['total', 'flux', 'heating', '(n,2n)']
        assert yamc.Tally(scores=with_basic).scores == with_basic

    def test_tally_all_reaction_types(self):
        """Test various ENDF reaction types are correctly parsed.

        All REACTION_MT lookups preserve the original string input.
        """
        reaction_tests = [
            '(n,3n)',
            '(n,4n)',
            '(n,t)',
            '(n,d)',
            '(n,3He)',
            '(n,na)',
            '(n,np)',
            '(n,2a)',
            '(n,nc)',
            '(n,fission)',
        ]

        for name in reaction_tests:
            tally = yamc.Tally(scores=[name])
            assert tally.scores == [name], f"Failed for {name}: expected {name}, got {tally.scores}"

    def test_tally_flux_score_constant(self):
        """Test that FLUX_SCORE constant is accessible."""
        assert yamc.Tally(scores=['flux']).scores == ['flux']

    def test_tally_name_and_id(self):
        """Test name and id set via constructor."""
        tally = yamc.Tally(name="Absorption Tally", id=42)
        assert tally.name == "Absorption Tally"
        assert tally.id == 42

        # Default values
        empty = yamc.Tally()
        assert empty.name is None
        assert empty.id is None

    def test_tally_repr(self):
        """Test tally string representation."""
        tally = yamc.Tally(scores=[101], name="Test Tally", id=1)

        repr_str = repr(tally)
        assert "Tally(" in repr_str
        assert "scores=[101]" in repr_str
        assert '"Test Tally"' in repr_str
        assert "id=1" in repr_str

    def test_tally_repr_production_scores(self):
        """Test tally string representation with production scores."""
        tally = yamc.Tally(scores=['H3-production', 101], name="Production Test", id=2)

        repr_str = repr(tally)
        assert "Tally(" in repr_str
        assert '"H3-production"' in repr_str
        assert "101" in repr_str

    def test_tally_repr_heating_scores(self):
        """Test tally string representation with heating scores."""
        tally = yamc.Tally(scores=['heating', 'heating-local'], name="Heating Test", id=3)

        repr_str = repr(tally)
        assert "Tally(" in repr_str
        assert '"heating"' in repr_str
        assert '"heating-local"' in repr_str

    def test_tally_repr_includes_units(self):
        """Test that tally repr includes derived units."""
        tally = yamc.Tally(scores=['flux', 'heating'])
        repr_str = repr(tally)
        assert 'units=[' in repr_str
        assert 'cm / source-particle' in repr_str
        assert 'eV / source-particle' in repr_str

    def test_tally_constructor_kwargs(self):
        """Test creating tally with constructor kwargs."""
        tally = yamc.Tally(scores=['flux'], name="My Tally", id=5)
        assert tally.scores == ['flux']
        assert tally.name == "My Tally"
        assert tally.id == 5

    def test_tally_constructor_with_mesh(self):
        """Test creating tally with mesh keyword in constructor."""
        mesh = yamc.RegularRectangularMesh([0, 0, 0], [1, 1, 1], [2, 2, 2])
        tally = yamc.Tally(scores=['flux'], mesh=mesh)
        assert tally.units == ['cm / cm³ / source-particle']


class TestTallyCellFilter:
    """Cell filters need cell ids, which the Geometry assigns."""

    def test_cells_before_geometry_raises_value_error(self):
        """A cell filter built before the Geometry is a normal Python error."""
        sphere = yamc.Sphere(radius=10.0)
        cell = yamc.Cell(region=sphere.below, name="ball")

        with pytest.raises(ValueError) as excinfo:
            yamc.Tally(scores=['flux'], cells=cell)

        message = str(excinfo.value)
        assert 'ball' in message
        assert 'Geometry' in message

    def test_set_cells_before_geometry_raises_value_error(self):
        """The cells setter follows the same rule as the constructor."""
        cell = yamc.Cell(region=yamc.Sphere(radius=10.0).below)
        tally = yamc.Tally(scores=['flux'])

        with pytest.raises(ValueError):
            tally.cells = cell

    def test_cells_after_geometry_works(self):
        """Building the Geometry first assigns ids, so the filter builds."""
        cell = yamc.Cell(region=yamc.Sphere(radius=10.0).below)
        yamc.Geometry(cells=[cell])

        tally = yamc.Tally(scores=['flux'], cells=cell)
        assert tally.cells == [cell.id]

    def test_explicit_cell_id_works_without_geometry(self):
        """An explicit Cell(id=...) is the other way to get an id."""
        cell = yamc.Cell(region=yamc.Sphere(radius=10.0).below, id=7)
        tally = yamc.Tally(scores=['flux'], cells=cell)
        assert tally.cells == [7]


class TestTallyUnits:
    """Test automatic unit derivation for tallies."""

    def test_flux_units(self):
        """Test flux tally gets cm / source-particle."""
        assert yamc.Tally(scores=['flux']).units == ['cm / source-particle']

    def test_heating_units(self):
        """Test heating tally gets eV / source-particle."""
        assert yamc.Tally(scores=['heating']).units == ['eV / source-particle']

    def test_heating_local_units(self):
        """Test heating-local tally gets eV / source-particle."""
        assert yamc.Tally(scores=['heating-local']).units == ['eV / source-particle']

    def test_damage_energy_units(self):
        """Test damage-energy tally gets eV / source-particle."""
        assert yamc.Tally(scores=['damage-energy']).units == ['eV / source-particle']

    def test_reaction_units(self):
        """Test reaction scores get reactions / source-particle."""
        for scores in (['total'], ['absorption'], ['fission'], [101]):
            assert yamc.Tally(scores=scores).units == ['reactions / source-particle']

    def test_production_units(self):
        """Test production scores get particles / source-particle."""
        for scores in (['H1-production'], ['He4-production']):
            assert yamc.Tally(scores=scores).units == ['particles / source-particle']

    def test_mixed_scores_units(self):
        """Test mixed scores each get their own units."""
        units = yamc.Tally(scores=['flux', 'heating', 101]).units
        assert units[0] == 'cm / source-particle'
        assert units[1] == 'eV / source-particle'
        assert units[2] == 'reactions / source-particle'

    def test_mesh_adds_cm3_denominator(self):
        """Test that mesh adds / cm³ to units."""
        mesh = yamc.RegularRectangularMesh([0, 0, 0], [1, 1, 1], [2, 2, 2])
        tally = yamc.Tally(scores=['flux'], mesh=mesh)
        assert tally.units == ['cm / cm³ / source-particle']

    def test_mesh_heating_units(self):
        """Test heating with mesh gets eV / cm³ / source-particle."""
        mesh = yamc.RegularRectangularMesh([0, 0, 0], [1, 1, 1], [2, 2, 2])
        tally = yamc.Tally(scores=['heating'], mesh=mesh)
        assert tally.units == ['eV / cm³ / source-particle']

    def test_energy_function_filter_units(self):
        """Test energy_function with user-supplied units."""
        energy = [1.0, 10.0, 100.0, 1000.0]
        y = [1.0, 2.0, 3.0, 4.0]

        tally = yamc.Tally(scores=['flux'], energy_function=(energy, y, "pSv·cm²"))
        assert tally.units == ['pSv·cm² · cm / source-particle']

    def test_energy_function_filter_no_units(self):
        """Test energy_function without user-supplied units falls back to score units."""
        energy = [1.0, 10.0, 100.0, 1000.0]
        y = [1.0, 2.0, 3.0, 4.0]

        tally = yamc.Tally(scores=['flux'], energy_function=(energy, y))
        assert tally.units == ['cm / source-particle']

    def test_two_scores_different_units(self):
        """Test that two different scores on the same tally get distinct units."""
        tally = yamc.Tally(scores=['flux', 'heating'])
        assert tally.units == [
            'cm / source-particle',
            'eV / source-particle',
        ]

    def test_two_scores_with_energy_function(self):
        """Test that energy_function combines with each score's base units."""
        energy = [1.0, 10.0, 100.0, 1000.0]
        y = [1.0, 2.0, 3.0, 4.0]

        tally = yamc.Tally(
            scores=['flux', 'heating'],
            energy_function=(energy, y, "pSv·cm²"))
        assert tally.units == [
            'pSv·cm² · cm / source-particle',
            'pSv·cm² · eV / source-particle',
        ]


class TestTallySimulation:
    """Test tallies in actual Monte Carlo simulation."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing tallies."""
        # Create sphere surface with vacuum boundary
        sphere = yamc.Sphere(
            x0=0.0,
            y0=0.0,
            z0=0.0,
            radius=2.0,
            boundary='vacuum')
        region = sphere.below
        
        # Create material with Li6
        material = yamc.Material(
            composition={"Li6": 1.0},
            density=5.5,
            temperature=294)
        material.read_nuclear_data({"Li6": "tests/Li6.arrow"})
        
        # Create cell
        cell = yamc.Cell(
            name="sphere_cell",
            region=region,
            material=material)
        geometry = yamc.Geometry(cells=[cell])
        
        # Create source
        source = yamc.NeutronSource(position=[0.0, 0.0, 0.0], direction=yamc.sources.Monodirectional(direction=[0.0, 0.0, 1.0]), energy=yamc.sources.Discrete([1e6], [1.0]))

        return geometry, source

    def test_simulation_with_absorption_tally(self, simple_model):
        """Test simulation with absorption tally."""
        geometry, source = simple_model

        absorption_tally = yamc.Tally(scores=[101], name="Absorption Tally")  # MT 101 = absorption
        tallies = [absorption_tally]
        
        # Create and run model
        model = yamc.Model(geometry=geometry, tallies=tallies, source=source)
        results = model.simulate_transport(total_particles=500)

        # Check absorption tally
        absorption_result = results[absorption_tally]
        assert absorption_tally.name == "Absorption Tally"
        # WelfordPerHistory default: n_batches == total_particles (one
        # sample per history). particles_per_chunk reports the transport
        # chunk size (derive_particles_per_chunk → total/10).
        assert absorption_result.n_batches == 500
        assert absorption_result.particles_per_chunk == 50

        # Statistics should be calculated
        if absorption_result.total_count[0] > 0:
            assert absorption_result.standard_deviation[0] >= 0.0
            assert absorption_result.relative_error[0] >= 0.0
    
    def test_simulation_with_multiple_tallies(self, simple_model):
        """Test simulation with multiple tallies."""
        geometry, source = simple_model

        absorption_tally = yamc.Tally(scores=[101], name="Absorption Events")
        elastic_tally = yamc.Tally(scores=[2], name="Elastic Scattering Events")

        tallies = [absorption_tally, elastic_tally]
        
        # Create and run model
        model = yamc.Model(geometry=geometry, tallies=tallies, source=source)
        results = model.simulate_transport(total_particles=500)

        # Check all tallies
        assert absorption_tally.name == "Absorption Events"
        assert elastic_tally.name == "Elastic Scattering Events"

        # WelfordPerHistory default: n_batches == total_particles.
        # particles_per_chunk is the transport chunk size (total/10).
        for tally in tallies:
            tally_result = results[tally]
            assert tally_result.n_batches == 500
            assert tally_result.particles_per_chunk == 50
            
    def test_simulation_without_user_tallies(self, simple_model):
        """Test simulation with only leakage tally (no user tallies)."""
        geometry, source = simple_model

        # Create model with no user tallies
        model = yamc.Model(geometry=geometry, tallies=[], source=source)
        model.simulate_transport(total_particles=500)

    def test_absorption_string_and_mt101_equivalence(self, simple_model):
        """Test that 'absorption' and MT 101 produce the same tally result."""
        geometry, source = simple_model

        # Create two tallies: one with string, one with integer
        str_tally = yamc.Tally(scores=['absorption'], name="Absorption String")
        int_tally = yamc.Tally(scores=[101], name="Absorption MT101")

        tallies = [str_tally, int_tally]

        model = yamc.Model(geometry=geometry, tallies=tallies, source=source)
        results = model.simulate_transport(total_particles=500)

        # Both should produce nearly identical mean and std_dev
        # (track-length f64 accumulation may differ at ULP level)
        import math
        str_result = results[str_tally]
        int_result = results[int_tally]
        for a, b in zip(str_result.mean, int_result.mean):
            assert math.isclose(a, b, rel_tol=1e-9), f"Means differ: {a} vs {b}"
        for a, b in zip(str_result.standard_deviation, int_result.standard_deviation):
            assert math.isclose(a, b, rel_tol=1e-9) or (a == 0.0 and b == 0.0), f"Std devs differ: {a} vs {b}"

        # But they should preserve their original input format
        assert str_tally.scores == ['absorption']
        assert int_tally.scores == [101]

    def test_tally_statistics_consistency(self, simple_model):
        """Test that tally statistics are consistent and reasonable."""
        geometry, source = simple_model

        absorption_tally = yamc.Tally(scores=[101], name="Statistics Test")
        tallies = [absorption_tally]
        
        # Run simulation
        model = yamc.Model(geometry=geometry, tallies=tallies, source=source)
        results = model.simulate_transport(total_particles=500)

        # Test statistics consistency
        absorption_result = results[absorption_tally]
        # WelfordPerHistory default: n_batches == total_particles.
        assert absorption_result.n_batches == 500

        # If we have results, test statistical relationships
        if absorption_result.total_count[0] > 0:
            # Mean should be positive
            assert absorption_result.mean[0] > 0

            # Relative error should be std_dev / mean (if mean > 0)
            if absorption_result.mean[0] > 0:
                expected_rel_error = absorption_result.standard_deviation[0] / absorption_result.mean[0]
                assert abs(absorption_result.relative_error[0] - expected_rel_error) < 1e-10


class TestFluxTally:
    """Test flux tallying functionality."""
    
    @pytest.fixture
    def flux_test_model(self):
        """Create a simple model for testing flux tallies."""
        # Create sphere surface with vacuum boundary
        sphere = yamc.Sphere(
            x0=0.0,
            y0=0.0,
            z0=0.0,
            radius=5.0,
            boundary='vacuum')
        region = sphere.below
        
        # Create material with Li6
        material = yamc.Material(
            composition={"Li6": 1.0},
            density=0.46,
            temperature=294)
        material.read_nuclear_data({"Li6": "tests/Li6.arrow"})
        
        # Create cell
        cell = yamc.Cell(
            name="sphere_cell",
            region=region,
            material=material)
        geometry = yamc.Geometry(cells=[cell])
        
        # Create source - point source at center
        source = yamc.NeutronSource(
            position=[0.0, 0.0, 0.0],
            energy=yamc.sources.Discrete([14.1e6], [1.0])  # 14.1 MeV
        )

        return geometry, source

    def test_flux_tally_using_string(self, flux_test_model):
        """Test flux tally using 'flux' string."""
        geometry, source = flux_test_model

        flux_tally = yamc.Tally(scores=['flux'], name="Flux Tally")
        tallies = [flux_tally]
        
        # Create and run model
        model = yamc.Model(geometry=geometry, tallies=tallies, source=source)
        results = model.simulate_transport(total_particles=10000)

        # Check flux tally results
        flux_result = results[flux_tally]
        assert flux_tally.name == "Flux Tally"
        # WelfordPerHistory default: n_batches == total_particles.
        # particles_per_chunk is the transport chunk size (total/10).
        assert flux_result.n_batches == 10000
        assert flux_result.particles_per_chunk == 1000
        assert len(flux_result.mean) == 1

        # Flux should be positive
        assert flux_result.mean[0] > 0.0

        # Should have statistics
        assert flux_result.standard_deviation[0] >= 0.0
        assert flux_result.relative_error[0] >= 0.0
    
    def test_flux_tally_using_constant(self, flux_test_model):
        """Test flux tally using FLUX_SCORE constant."""
        geometry, source = flux_test_model

        flux_tally = yamc.Tally(scores=['flux'], name="Flux with Constant")
        tallies = [flux_tally]
        
        # Create and run model
        model = yamc.Model(geometry=geometry, tallies=tallies, source=source)
        results = model.simulate_transport(total_particles=10000)

        # Check results
        assert results[flux_tally].mean[0] > 0.0

    def test_flux_and_reaction_tally_mixed(self, flux_test_model):
        """Test tally with both flux and reaction scores."""
        geometry, source = flux_test_model

        # flux and absorption
        tally = yamc.Tally(scores=['flux', 101], name="Mixed Tally")
        tallies = [tally]
        
        # Create and run model
        model = yamc.Model(geometry=geometry, tallies=tallies, source=source)
        results = model.simulate_transport(total_particles=10000)

        # Should have two scores
        result = results[tally]
        assert len(result.mean) == 2
        assert tally.scores == ['flux', 101]

        # Flux (index 0) should be positive
        assert result.mean[0] > 0.0

        # Absorption might be zero or positive
        assert result.mean[1] >= 0.0


class TestTallyIntegration:
    """Integration tests for tally system."""
    
    def test_tally_display_output(self):
        """Test that tally display output is reasonable."""
        tally = yamc.Tally(scores=[101], name="Test Display")

        # Test string output doesn't crash
        str_output = str(tally)
        assert "Test Display" in str_output
        assert "Mean:" in str_output
        
    def test_model_constructor_with_tallies(self):
        """Test that model accepts tallies in constructor."""
        # Create minimal geometry
        sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0, boundary='vacuum')
        region = sphere.below
        material = yamc.Material(
            composition={"Li6": 1.0},
            density=1.0,
            temperature=294)
        material.read_nuclear_data({"Li6": "tests/Li6.arrow"})
        cell = yamc.Cell(name="test", region=region, material=material)
        geometry = yamc.Geometry(cells=[cell])

        # Create source
        source = yamc.NeutronSource(position=[0.0, 0.0, 0.0], direction=yamc.sources.Monodirectional([0.0,0.0,1.0]), energy=yamc.sources.Discrete([1e6], [1.0]))

        tally = yamc.Tally(scores=[101])
        tallies = [tally]

        # Model should accept tallies
        model = yamc.Model(geometry=geometry, tallies=tallies, source=source)
        assert model is not None
        
        # Should be able to access tallies
        assert len(model.tallies) == 1
        assert model.tallies[0].scores == [101]


class TestTallyNuclides:
    """Test per-nuclide tally functionality."""

    def test_nuclides_default_empty(self):
        """Default tally has empty nuclides list."""
        tally = yamc.Tally()
        assert tally.nuclides == []

    def test_nuclides_kwarg(self):
        """Can set nuclides list via constructor kwarg."""
        tally = yamc.Tally(nuclides=['Li6', 'Li7', 'Be9'])
        assert tally.nuclides == ['Li6', 'Li7', 'Be9']

    def test_nuclides_with_total(self):
        """'total' keyword round-trips correctly."""
        tally = yamc.Tally(nuclides=['Li6', 'total'])
        assert tally.nuclides == ['Li6', 'total']

    def test_nuclides_with_scores(self):
        """Can pass nuclides alongside scores in Tally constructor."""
        tally = yamc.Tally(
            scores=['H3-production'],
            nuclides=['Li6', 'Li7'])
        assert tally.nuclides == ['Li6', 'Li7']
        assert tally.scores == ['H3-production']

    def test_nuclide_tally_h3_production(self):
        """Per-nuclide H3-production sum matches total within statistics."""
        # Simple Li6+Li7+Be9 sphere with 14 MeV source
        sphere = yamc.Sphere(radius=200.0, boundary='vacuum')
        material = yamc.Material(
            composition={"Li6": 0.035, "Li7": 0.465, "Be9": 0.5},
            density=2.0,
            temperature=294)
        material.read_nuclear_data({
            "Be9": "tests/Be9.arrow",
            "Li6": "tests/Li6.arrow",
            "Li7": "tests/Li7.arrow",
        })
        cell = yamc.Cell(region=sphere.below, material=material)
        geometry = yamc.Geometry([cell])

        source = yamc.NeutronSource(
            energy=yamc.sources.Discrete([14.06e6], [1.0]),
            position=(0, 0, 0))
        # Tally A: total
        tally_total = yamc.Tally(scores=['H3-production'], cells=cell,
                               name="total")

        # Tally B: per-nuclide
        tally_nuc = yamc.Tally(scores=['H3-production'],
                             nuclides=['Li6', 'Li7', 'Be9'],
                             cells=cell, name="per_nuc")

        model = yamc.Model(geometry=geometry, source=source,
                         tallies=[tally_total, tally_nuc])
        results = model.simulate_transport(total_particles=30000, seed=42)

        total_mean = results[tally_total].mean[0]
        nuc_mean = results[tally_nuc].mean
        per_nuc_sum = sum(nuc_mean)

        # Sum of per-nuclide should match total to machine precision
        # (same scoring path, deterministic)
        assert abs(per_nuc_sum - total_mean) < 1e-10 * abs(total_mean), \
            f"per_nuc_sum={per_nuc_sum}, total={total_mean}"

        # Li6 has a large (n,t) cross section at 14 MeV
        assert nuc_mean[0] > 0, "Li6 H3-production should be > 0"
        # Li7 also contributes
        assert nuc_mean[1] > 0, "Li7 H3-production should be > 0"

    def test_nuclide_tally_with_total_keyword(self):
        """'total' nuclide bin matches the no-nuclide tally."""
        sphere = yamc.Sphere(radius=200.0, boundary='vacuum')
        material = yamc.Material(
            composition={"Li6": 0.035, "Li7": 0.465, "Be9": 0.5},
            density=2.0,
            temperature=294)
        material.read_nuclear_data({
            "Be9": "tests/Be9.arrow",
            "Li6": "tests/Li6.arrow",
            "Li7": "tests/Li7.arrow",
        })
        cell = yamc.Cell(region=sphere.below, material=material)
        geometry = yamc.Geometry([cell])

        source = yamc.NeutronSource(
            energy=yamc.sources.Discrete([14.06e6], [1.0]),
            position=(0, 0, 0))
        tally_no_nuc = yamc.Tally(scores=['H3-production'], cells=cell)
        tally_with_total = yamc.Tally(scores=['H3-production'],
                                     nuclides=['Li6', 'total'],
                                     cells=cell)

        model = yamc.Model(geometry=geometry, source=source,
                         tallies=[tally_no_nuc, tally_with_total])
        results = model.simulate_transport(total_particles=10000, seed=7)

        no_nuc_mean = results[tally_no_nuc].mean[0]
        # "total" is the second nuclide bin (index 1)
        total_bin_mean = results[tally_with_total].mean[1]

        assert abs(total_bin_mean - no_nuc_mean) < 1e-10 * abs(no_nuc_mean), \
            f"total_bin={total_bin_mean}, no_nuc={no_nuc_mean}"