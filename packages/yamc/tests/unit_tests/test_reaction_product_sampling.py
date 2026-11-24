"""
Test suite for reaction product sampling functionality.

Tests both the Rust implementation via PyO3 bindings and
the Python interface for nuclear reaction product sampling.
"""

import statistics

import pytest
import yamc


class TestTabulatedSampling:
    """Test tabulated distribution sampling"""
    
    def test_tabulated_creation(self):
        """Test creating a tabulated distribution"""
        x = [-1.0, 0.0, 1.0]
        p = [0.2, 0.6, 1.0]  # CDF values
        
        tab = yamc.Tabulated(x, p)
        
        assert tab.values == x
        assert tab.probabilities == p
    
    def test_tabulated_sampling(self):
        """Sampling reproduces the distribution `Tabulated` actually builds.

        With no explicit CDF, `p` is a PDF and the CDF comes from trapezoidal
        integration, so `p = [0.0, 0.5, 1.0]` over `x = [-1, 0, 1]` is a rising
        density, NOT the uniform distribution this test used to claim: the bin
        weights are 0.25 for [-1, 0] and 0.75 for [0, 1], giving a 3:1 split
        about zero and a mean near +0.17.

        The old assertion (`abs(mean) < 0.2` over 1000 samples) checked that
        wrong expectation and sat only ~3 sigma from failing, which is why it
        flaked on Windows CI at 0.2005. This asserts the real properties with
        enough samples that neither bound is reachable by chance.
        """
        x = [-1.0, 0.0, 1.0]
        p = [0.0, 0.5, 1.0]

        tab = yamc.Tabulated(x, p)

        n = 20_000
        samples = [tab.sample() for _ in range(n)]

        assert all(-1.0 <= s <= 1.0 for s in samples)

        # Bin weights: 3x as many samples in [0, 1] as in [-1, 0]. Binomial
        # noise on the fraction is sqrt(0.75 * 0.25 / n) ~ 0.003 at this n, so
        # the ratio's sigma is ~0.05 and these bounds are ~6 sigma out.
        right = sum(1 for s in samples if s >= 0.0)
        left = n - right
        assert left > 0, "no samples in the low-weight bin; sampler is broken"
        ratio = right / left
        assert 2.7 < ratio < 3.3, (
            f"expected a 3:1 split about zero from the trapezoidal CDF, got {ratio:.3f}"
        )

        # Mean of that density is ~0.1705; sigma of the sample mean is ~0.004 at
        # this n, so this band is ~8 sigma wide either side.
        mean = statistics.mean(samples)
        assert 0.14 < mean < 0.20, f"mean {mean:.4f} outside the expected band"


class TestIsotropicSampling:
    """Test isotropic scattering utilities"""
    
    def test_sample_scatter_cosine(self):
        """Test isotropic mu sampling"""
        samples = [yamc.sample_scatter_cosine() for _ in range(1000)]
        
        # All samples should be valid cosines
        assert all(-1.0 <= s <= 1.0 for s in samples)
        
        # Should be roughly uniform (mean ~ 0)
        mean = statistics.mean(samples)
        assert abs(mean) < 0.1


class TestReactionProduct:
    """Test ReactionProduct sampling"""
    
    def test_create_test_reaction_product(self):
        """Test creating a test reaction product"""
        product = yamc.create_test_reaction_product()
        
        assert product.particle == "neutron"
        assert product.emission_mode == "prompt"
        assert product.get_decay_rate() == 0.0
        assert product.num_distributions > 0
    
    def test_reaction_product_sampling(self):
        """Test sampling from reaction product"""
        product = yamc.create_test_reaction_product()

        incoming_energy = 14e6  # 14 MeV

        # Sample multiple times
        results = [product.sample(incoming_energy) for _ in range(100)]

        for e_out, mu in results:
            # Energy should be ~99% of incoming (LevelInelastic with mass_ratio=0.99)
            assert abs(e_out - 0.99 * incoming_energy) < 1e-6

            # Mu should be valid cosine
            assert -1.0 <= mu <= 1.0
    
    def test_reaction_product_properties(self):
        """Test reaction product property methods"""
        product = yamc.create_test_reaction_product()
        
        # Test particle type checking
        assert product.is_particle_type("neutron")
        assert not product.is_particle_type("photon")
        
        # Test emission mode
        assert product.is_prompt()
        assert not product.is_delayed()
    
    def test_multiple_sampling(self):
        """Test sampling multiple particles from one product"""
        product = yamc.create_test_reaction_product()
        
        results = product.sample_multiple(14e6)
        
        # Should get at least one result
        assert len(results) >= 1
        
        for e_out, mu in results:
            assert e_out > 0
            assert -1.0 <= mu <= 1.0


class TestSamplingStatistics:
    """Test statistical properties of sampling"""
    
    def test_angular_distribution_statistics(self):
        """Test that angular distributions have correct statistical properties"""
        product = yamc.create_test_reaction_product()
        
        # Sample many mu values
        n_samples = 10000
        mu_samples = [product.sample(14e6)[1] for _ in range(n_samples)]
        
        # Check mean and standard deviation are reasonable for the test distribution
        mean_mu = statistics.mean(mu_samples)
        std_mu = statistics.stdev(mu_samples)
        
        # The test distribution should be roughly centered and not too narrow
        assert abs(mean_mu) < 0.1  # Roughly isotropic
        assert std_mu > 0.3       # Not too peaked
    
    def test_energy_distribution(self):
        """Test energy distribution for inelastic-like scattering"""
        product = yamc.create_test_reaction_product()

        energies = [1e5, 1e6, 14e6, 1e8]  # Various incoming energies

        for incoming_energy in energies:
            e_out, mu = product.sample(incoming_energy)

            # For the test product (LevelInelastic with mass_ratio=0.99), e_out = 0.99 * e_in
            expected_e_out = 0.99 * incoming_energy
            assert abs(e_out - expected_e_out) / expected_e_out < 1e-10
    
    def test_sampling_reproducibility_with_seed(self):
        """Test that sampling is reproducible when using the same conditions"""
        # Note: Python sampling uses thread_rng internally, so we can't easily seed it
        # This test just checks that sampling produces different results on repeated calls
        product = yamc.create_test_reaction_product()
        
        results1 = [product.sample(14e6) for _ in range(10)]
        results2 = [product.sample(14e6) for _ in range(10)]
        
        # Results should be different (very unlikely to be identical)
        assert results1 != results2


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_tabulated_distribution(self):
        """Test handling of empty distributions"""
        # Empty arrays should not crash
        empty_tab = yamc.Tabulated([], [])
        
        # Should return 0.0 for empty distribution
        result = empty_tab.sample()
        assert result == 0.0
    
    def test_single_point_distribution(self):
        """Test single-point distribution"""
        single_tab = yamc.Tabulated([0.5], [1.0])
        
        # Should always return the single value
        for _ in range(10):
            assert single_tab.sample() == 0.5


class TestPhysicsValidation:
    """Test that sampling results are physically reasonable"""
    
    def test_mu_bounds(self):
        """Test that all mu values are valid cosines"""
        product = yamc.create_test_reaction_product()
        
        # Test at various energies
        energies = [10 ** (4 + i * 4 / 19) for i in range(20)]  # 10 keV to 100 MeV
        
        for energy in energies:
            for _ in range(10):  # Multiple samples per energy
                e_out, mu = product.sample(energy)
                
                # Mu must be a valid cosine
                assert -1.0 <= mu <= 1.0, f"Invalid mu={mu} at E={energy}"
    
    def test_energy_positivity(self):
        """Test that outgoing energies are positive"""
        product = yamc.create_test_reaction_product()
        
        energies = [1e3, 1e6, 14e6, 1e8]
        
        for energy in energies:
            for _ in range(10):
                e_out, mu = product.sample(energy)
                assert e_out > 0, f"Non-positive energy {e_out} at E={energy}"


class TestIntegration:
    """Integration tests with real nuclear data (if available)"""
    
    def test_with_nuclide_data(self):
        """Test sampling integration with nuclide data"""
        # This would test integration with actual nuclide reactions
        # For now, just test that our interfaces are compatible
        
        product = yamc.create_test_reaction_product()
        
        # Simulate typical neutron transport energies
        transport_energies = [
            2.53e-8,  # Thermal
            1e-6,     # Epithermal
            1e-3,     # Intermediate
            1e0,      # Fast
            14e6,     # Fusion
        ]
        
        for energy in transport_energies:
            e_out, mu = product.sample(energy)
            
            # Basic physics checks
            assert e_out > 0
            assert -1.0 <= mu <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])