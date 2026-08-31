# API reference

## Conversion functions

High-level entry points that accept source files (ACE or ENDF) and write
simulation-ready `.arrow/` output.

::: nuclear_data_to_arrow.convert_neutron

::: nuclear_data_to_arrow.convert_photon

::: nuclear_data_to_arrow.convert_transmutation

::: nuclear_data_to_arrow.convert_branching

## Low-level writers

Accept endf-python objects directly.  Useful when you have already loaded
data through your own pipeline.

::: nuclear_data_to_arrow.neutron_writer.export_neutron_to_arrow

::: nuclear_data_to_arrow.photon_writer.export_photon_to_arrow

::: nuclear_data_to_arrow.transmutation_writer.export_transmutation_to_arrow

## Readers

Read `.arrow/` directories back into Python dicts for inspection or
verification.

::: nuclear_data_to_arrow.neutron_reader.read_neutron_from_arrow

::: nuclear_data_to_arrow.photon_reader.read_photon_from_arrow

## Verification

Compare an endf-python object against its Arrow export.  Uses tolerance-based
comparison (`np.allclose` with `rtol=1e-12`) and verifies synthesized MTs
and FastXSGrid consistency.

::: nuclear_data_to_arrow.verify.verify_neutron

::: nuclear_data_to_arrow.verify.verify_photon

## Synthesis module

MT synthesis and FastXSGrid construction algorithms.

::: nuclear_data_to_arrow.synthesis.is_scattering_mt

::: nuclear_data_to_arrow.synthesis.is_fission_mt

::: nuclear_data_to_arrow.synthesis.synthesize_hierarchical_mts

::: nuclear_data_to_arrow.synthesis.build_fast_xs

## Constants

::: nuclear_data_to_arrow.synthesis.N_LOG_BINS

::: nuclear_data_to_arrow.synthesis.SYNTHETIC_MTS

::: nuclear_data_to_arrow.synthesis.ALL_SCATTERING_MTS

::: nuclear_data_to_arrow.synthesis.FISSION_MTS

::: nuclear_data_to_arrow.synthesis.ABSORPTION_MTS
