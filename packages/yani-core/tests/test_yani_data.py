"""yani.data exposes the same TICE 2013 abundance records as yamc.data.

The binding is shared with yamc, but yani's wheel registers its own copy, so
this checks the registration on this wheel rather than the function itself.
"""
import pytest

yani = pytest.importorskip("yani", reason="standalone yani wheel not installed")


def test_natural_abundance_records_match_natural_abundance():
    records = yani.data.natural_abundance_records()
    assert records.keys() == yani.data.natural_abundance().keys()
    assert records["Fe58"]["representative_value"] == 0.00282
    assert records["Fe58"]["representative_uncertainty"] == 0.00012
    assert records["Fe58"]["best_measurement_uncertainty"] == 0.000027
