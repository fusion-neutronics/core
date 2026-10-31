#!/usr/bin/env python3
"""
Example: Using particle tracking to debug transport behavior.

This script demonstrates how to:
1. Run a simulation with particle tracking enabled
2. Convert tracks to a pandas DataFrame for analysis
3. Filter events by type, energy, distribution, etc.
4. Trace a particle's history through the geometry
"""

import yamc
import pandas as pd

# =============================================================================
# Setup a simple model for demonstration
# =============================================================================

# Create a simple geometry - sphere of iron
iron = yamc.Material(
    composition={"Fe56": 1.0},
    density=7.874,
    temperature=294)
# Use test data file that ships with the repo
iron.read_nuclear_data({'Fe56': "tests/Fe56.arrow"})

sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
cell = yamc.Cell(region=sphere.below, material=iron)

geometry = yamc.Geometry([cell])

# Source at center, 1 MeV neutrons
source = yamc.NeutronSource()
source.position = (0.0, 0.0, 0.0)
source.energy = 1.0e6

model = yamc.Model(geometry=geometry, source=source)

# =============================================================================
# Run with particle tracking
# =============================================================================

print("Running simulation with particle tracking...")

# Track the first 50 histories (recommended for debugging - keeps memory manageable).
# capture_tracks also accepts a range (e.g. range(20, 30) or range(0, 1000, 10)),
# an explicit list of history indices ([20, 30, 45]), or 'all'.
results = model.simulate_transport(total_particles=100, capture_tracks=50)
tracks = results.tracks

# Or track all particles (warning: can use lots of memory for large simulations)
# tracks = model.simulate_transport(total_particles=100, capture_tracks='all').tracks


print(f"Tracked {len(tracks)} particle histories")
print(f"Total events: {tracks.total_events()}")

# =============================================================================
# Convert to DataFrame for easy analysis
# =============================================================================

df = pd.DataFrame(tracks.to_dataframe_records())
print(f"\nDataFrame shape: {df.shape}")
print(f"\nEvent types:\n{df['event_type'].value_counts()}")

# =============================================================================
# Example 1: Find particles that end up at a specific energy
# =============================================================================

# Find collisions that result in energy around 500 keV
target_energy = 5.0e5  # eV
tolerance = 0.5e5  # +/- 50 keV

suspicious = df[
    (df['event_type'] == 'collision') &
    (df['energy_out'] > target_energy - tolerance) &
    (df['energy_out'] < target_energy + tolerance)
]

print("\n=== Collisions resulting in ~500 keV ===")
print(f"Found {len(suspicious)} events")
if len(suspicious) > 0:
    print(suspicious[['particle_id', 'nuclide', 'reaction_mt', 'distribution',
                      'energy_dist', 'energy_in', 'energy_out']].head(10))

# =============================================================================
# Example 2: Trace a specific particle's history
# =============================================================================

# Get a particle that had multiple collisions
collision_counts = df[df['event_type'] == 'collision'].groupby('particle_id').size()
if len(collision_counts) > 0:
    # Find particle with most collisions
    particle_id = collision_counts.idxmax()

    # Get all events for this particle
    history = df[df['particle_id'] == particle_id].sort_index()

    print(f"\n=== Full history for particle {particle_id} ({len(history)} events) ===")
    for _, event in history.iterrows():
        print(f"  {event['event_type']:20s} E={event['energy_in']:.4e} -> {event['energy_out']:.4e} eV "
              f"at ({event['x']:.3f}, {event['y']:.3f}, {event['z']:.3f})")
        if event['event_type'] == 'collision':
            print(f"      {event['nuclide']} MT{event['reaction_mt']} via {event['distribution']}")
            if pd.notna(event.get('energy_dist')):
                print(f"      Energy dist: {event['energy_dist']}")

# =============================================================================
# Example 3: Analyze which distributions are being used
# =============================================================================

collisions = df[df['event_type'] == 'collision']
if len(collisions) > 0:
    print("\n=== Distribution usage ===")
    print(collisions['distribution'].value_counts())

    if 'energy_dist' in collisions.columns:
        print("\n=== Energy distribution usage (within UncorrelatedAngleEnergy) ===")
        uncorr = collisions[collisions['distribution'] == 'UncorrelatedAngleEnergy']
        if len(uncorr) > 0:
            print(uncorr['energy_dist'].value_counts())

# =============================================================================
# Example 4: Follow particle genealogy (parent-child relationships)
# =============================================================================

# Find secondary particles (generation > 0)
secondaries = df[df['generation'] > 0]
print("\n=== Secondary particles ===")
print(f"Found {secondaries['particle_id'].nunique()} secondary particles")

if len(secondaries) > 0:
    births = secondaries[secondaries['event_type'].str.contains('birth')]
    if len(births) > 0:
        print(f"Birth reactions:\n{births['birth_reaction'].value_counts()}")

# =============================================================================
# Example 5: Energy statistics
# =============================================================================

if len(collisions) > 0:
    print("\n=== Collision energy statistics ===")
    print(f"Incoming energy range: {collisions['energy_in'].min():.4e} - {collisions['energy_in'].max():.4e} eV")
    print(f"Outgoing energy range: {collisions['energy_out'].min():.4e} - {collisions['energy_out'].max():.4e} eV")
    print(f"Mean energy loss per collision: {(collisions['energy_in'] - collisions['energy_out']).mean():.4e} eV")

# =============================================================================
# Example 6: Reaction breakdown by nuclide
# =============================================================================

if len(collisions) > 0:
    print("\n=== Reactions by nuclide ===")
    for nuclide in collisions['nuclide'].unique():
        if pd.notna(nuclide):
            nuc_collisions = collisions[collisions['nuclide'] == nuclide]
            print(f"\n{nuclide}:")
            print(f"  Total collisions: {len(nuc_collisions)}")
            print(f"  MT numbers: {nuc_collisions['reaction_mt'].value_counts().to_dict()}")

print("\nDone!")
