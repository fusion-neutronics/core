"""Integration test: 27-cell void grid vs 3x3x3 mesh tally.

Creates a 3x3x3 grid of void cells from planes with a photon point source
at the center. Scores flux in each cell and on a matching 3x3x3 regular mesh.
Verifies that cell flux and mesh voxel flux agree for every bin.

This tests mesh tally track-length scoring including:
- Tracks starting inside each voxel
- Tracks entering a voxel from outside (mesh entry logic)
- Tracks exiting through lower mesh boundaries
"""

import pytest
import yamc


def _build_27cell_model(total_particles=250000, seed=42):
    """Build a 3x3x3 void grid model with cell and mesh tallies."""
    # Grid: 3 bins in each dimension, from -3 to 3 (bin width = 2 cm)
    LO, HI, N = -3.0, 3.0, 3
    step = (HI - LO) / N
    boundaries = [LO + i * step for i in range(N + 1)]  # [-3, -1, 1, 3]

    # Create bounding planes (vacuum on the outer faces)
    xplanes, yplanes, zplanes = [], [], []
    for i, b in enumerate(boundaries):
        bt = "vacuum" if (i == 0 or i == N) else "transmission"
        xplanes.append(yamc.Plane(axis="x", offset=b, boundary=bt))
        yplanes.append(yamc.Plane(axis="y", offset=b, boundary=bt))
        zplanes.append(yamc.Plane(axis="z", offset=b, boundary=bt))

    # Build 27 void cells
    cells = []
    for iz in range(N):
        for iy in range(N):
            for ix in range(N):
                region = (
                    xplanes[ix].above & xplanes[ix + 1].below
                    & yplanes[iy].above & yplanes[iy + 1].below
                    & zplanes[iz].above & zplanes[iz + 1].below
                )
                cell = yamc.Cell(
                    name=f"cell_{ix}_{iy}_{iz}",
                    region=region)
                cells.append(cell)

    geometry = yamc.Geometry(cells)

    # Photon point source at origin, isotropic, 1 MeV
    source = yamc.PhotonSource(
        position=(0, 0, 0),
        energy=yamc.sources.Discrete([1e6], [1.0]))

    # 27 cell tallies (one per cell)
    cell_tallies = []
    for cell in cells:
        t = yamc.Tally(
            scores=["flux"],
            name=f"cell_{cell.id}",
            cells=cell,
            particle="photon")
        cell_tallies.append(t)

    # One mesh tally covering the entire geometry
    mesh = yamc.RegularRectangularMesh(
        lower_left=[LO, LO, LO],
        upper_right=[HI, HI, HI],
        shape=[N, N, N])
    mesh_tally = yamc.Tally(
        scores=["flux"],
        name="mesh_flux",
        particle="photon",
        mesh=mesh)

    all_tallies = cell_tallies + [mesh_tally]
    model = yamc.Model(geometry=geometry, tallies=all_tallies, source=source,
                     transport_secondary_photons=True)
    return model, cells, cell_tallies, mesh_tally, total_particles, seed


def test_mesh_vs_cell_flux_void_27cells():
    """Each mesh voxel flux should match the corresponding cell flux."""
    model, cells, cell_tallies, mesh_tally, total_particles, seed = _build_27cell_model()
    results = model.simulate_transport(total_particles=total_particles, seed=seed)

    mesh_means = list(results[mesh_tally].mean)
    assert len(mesh_means) == 27, f"Expected 27 mesh bins, got {len(mesh_means)}"

    N = 3
    max_rel_diff = 0.0
    mismatches = []

    for iz in range(N):
        for iy in range(N):
            for ix in range(N):
                flat_idx = iz * N * N + iy * N + ix
                cell_tally = cell_tallies[flat_idx]
                cell_mean = results[cell_tally].mean
                if isinstance(cell_mean, list):
                    cell_mean = cell_mean[0]

                mesh_mean = mesh_means[flat_idx]

                if cell_mean == 0.0 and mesh_mean == 0.0:
                    continue

                ref = max(abs(cell_mean), abs(mesh_mean))
                rel_diff = abs(cell_mean - mesh_mean) / ref if ref > 0 else 0

                max_rel_diff = max(max_rel_diff, rel_diff)

                if rel_diff > 0.01:  # > 1% difference
                    mismatches.append(
                        f"  ({ix},{iy},{iz}): cell={cell_mean:.4e} mesh={mesh_mean:.4e} "
                        f"diff={rel_diff*100:.2f}%"
                    )

    if mismatches:
        msg = f"Cell vs mesh flux mismatches (max {max_rel_diff*100:.2f}%):\n"
        msg += "\n".join(mismatches)
        pytest.fail(msg)

    # Also check that the totals match
    cell_total = sum(
        results[ct].mean[0] if isinstance(results[ct].mean, list) else results[ct].mean
        for ct in cell_tallies
    )
    mesh_total = sum(mesh_means)

    if cell_total > 0:
        total_rel_diff = abs(cell_total - mesh_total) / cell_total
        assert total_rel_diff < 0.001, (
            f"Total flux mismatch: cell={cell_total:.6e} mesh={mesh_total:.6e} "
            f"diff={total_rel_diff*100:.3f}%"
        )


def test_mesh_vs_cell_symmetry():
    """Opposing cells should have approximately equal flux (by symmetry)."""
    model, cells, cell_tallies, mesh_tally, total_particles, seed = _build_27cell_model(
        total_particles=1000000
    )
    results = model.simulate_transport(total_particles=total_particles, seed=seed)

    N = 3
    # Check x-symmetry: cell (0,y,z) vs cell (2,y,z) should be similar
    for iz in range(N):
        for iy in range(N):
            idx_lo = iz * N * N + iy * N + 0
            idx_hi = iz * N * N + iy * N + 2
            lo = results[cell_tallies[idx_lo]].mean
            hi = results[cell_tallies[idx_hi]].mean
            if isinstance(lo, list):
                lo = lo[0]
            if isinstance(hi, list):
                hi = hi[0]
            if lo > 0 and hi > 0:
                ratio = lo / hi
                # Should be within ~10% of 1.0 (statistical, void = no asymmetry)
                assert 0.8 < ratio < 1.2, (
                    f"Symmetry broken: cell({0},{iy},{iz})={lo:.4e} vs "
                    f"cell({2},{iy},{iz})={hi:.4e}, ratio={ratio:.3f}"
                )
