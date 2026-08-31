import yamc

# Configure cross section data paths
yamc.cross_section_data = {
    "Be9": "tests/Be9.arrow",
    "Fe54": "tests/Fe54.arrow",
    "Fe56": "tests/Fe56.arrow",
    "Fe57": "tests/Fe57.arrow",
    "Fe58": "tests/Fe58.arrow",
    "Li6": "tests/Li6.arrow",
    "Li7": "tests/Li7.arrow",
    "Pb208": "tests/Pb208.arrow",  # Added for product examples
}

print("=== Basic Nuclide Properties ===")
nuc1 = yamc.Nuclide('Li6')
print(f"Nuclide: {nuc1.name}")
print(f"Element: {nuc1.element}")
print(f"Atomic number: {nuc1.atomic_number}")
print(f"Mass number: {nuc1.mass_number}")
print(f"Available temperatures: {nuc1.available_temperatures}")
reaction_mts = nuc1.reaction_mts
if reaction_mts:
    print(f"Available reaction MTs: {reaction_mts[:10]}...")  # Show first 10
else:
    print("No reaction MTs available")

print("\n=== Cross Section Data ===")
# Get cross section data using both MT number and reaction name
micro_n_gamma, energy = nuc1.microscopic_cross_section(reaction='(n,gamma)', temperature='294')
micro_mt_3, energy = nuc1.microscopic_cross_section(reaction=3, temperature='294')

print(f"(n,gamma) cross section at 1 eV: {micro_n_gamma[0]:.6f} barns")
print(f"MT 3 cross section at 1 eV: {micro_mt_3[0]:.6f} barns")
print(f"Energy grid points: {len(energy)}")

print("\n=== Reaction Products and Angular Distributions ===")
# Load a heavier nuclide that has product data
nuc_pb = yamc.Nuclide('Pb208')
nuc_pb.read_nuclear_data('tests/Pb208.arrow')
reactions = nuc_pb.reactions['294']

# Find reactions with product data
reactions_with_products = []
for mt, reaction in reactions.items():
    reactions_with_products.append((mt, reaction))

print(f"Found {len(reactions_with_products)} reactions with product data")

# Examine the first few reactions with products
for i, (mt, reaction) in enumerate(reactions_with_products[:5]):
    if len(reaction.products) > 0:
        print(f"\n--- Reaction MT {mt} ---")
        product = reaction.products[0]
        print(f"  Particle: {product.particle}")
        print(f"  Emission mode: {product.emission_mode}")
        print(f"  Decay rate: {product.get_decay_rate()} /s")
        print(f"  Number of distributions: {product.num_distributions}")

        # We can now examine distribution types through the Python wrapper
        try:
            dist_types = product.distribution_types
            print(f"  Distribution types: {dist_types}")
        except Exception as e:
            print(f"  Error getting distribution types: {e}")

        # Sample from the reaction product (only for neutrons with energy distributions)
        # Note: Some products (like elastic MT2) only have angle distributions and use
        # two-body kinematics for energy, so direct sampling may not be supported.
        if product.particle == "neutron":
            try:
                outgoing_energy, mu = product.sample(1e6)  # 1 MeV incoming
                print(f"  Sample result at 1 MeV: E_out = {outgoing_energy:.2e} eV, mu = {mu:.3f}")
            except:  # noqa: E722 - need bare except to catch Rust panics
                # Catch both regular exceptions and Rust panics (PanicException)
                print("  Sampling not supported for this product (likely elastic/two-body kinematics)")

        # Note: The following demonstrates full distribution examination
        # which is now possible through our enhanced Python wrapper
        # The distribution_types property shows what types of distributions are available
        # Currently supported types:
        # - 'UncorrelatedAngleEnergy': Independent angle and energy sampling
        # - 'KalbachMann': Correlated angle-energy distribution with Kalbach-Mann systematics
        
        # For detailed distribution examination, we can check the types
        if hasattr(product, 'distribution_types'):
            for j, dist_type in enumerate(product.distribution_types):
                print(f"  Distribution {j+1}: {dist_type}")
                if dist_type == 'UncorrelatedAngleEnergy':
                    print("    -> Independent angle and energy sampling")
                elif dist_type == 'KalbachMann':
                    print("    -> Correlated Kalbach-Mann distribution")
                else:
                    print(f"    -> Unknown distribution type: {dist_type}")

print("\n=== Reaction Sampling ===")
# Demonstrate reaction sampling (Monte Carlo physics)
print("Sampling reactions at different energies:")
energies_to_test = [1e-3, 1.0, 1e6, 14.1e6]  # thermal, eV, MeV, 14.1 MeV

for energy in energies_to_test:
    reaction = nuc1.sample_reaction(energy=energy, temperature='294', seed=42)
    if reaction:
        print(f"  Energy {energy:.1e} eV -> MT {reaction['mt_number']}")
    else:
        print(f"  Energy {energy:.1e} eV -> No reaction sampled")

print("\n=== Product Sampling Demonstration ===")
# Demonstrate our new sampling functionality with a simple test product
print("Creating and sampling from reaction products:")

# Create test reaction products using our Python API
test_product = yamc.create_test_reaction_product()
print(f"Test product particle type: {test_product.particle}")

# Sample from it at different energies
for energy in [1e5, 1e6, 5e6]:
    try:
        e_out, mu = test_product.sample(energy)
        print(f"  Input: {energy:.0e} eV -> Output: E={e_out:.2e} eV, mu={mu:.3f}")
    except Exception as e:
        print(f"  Sampling at {energy:.0e} eV failed: {e}")

# Demonstrate sampling functions
print("\nSampling functions:")
mu_isotropic = yamc.sample_scatter_cosine()
print(f"Isotropic mu sample: {mu_isotropic:.3f}")

# Test tabulated sampling
x_vals = [-1.0, 0.0, 1.0]
p_vals = [0.0, 0.5, 1.0]  # CDF
mu_tabulated = yamc.Tabulated(x_vals, p_vals).sample()
print(f"Tabulated mu sample: {mu_tabulated:.3f}")
