# Prepared-data directory

This directory is populated by `scripts/prepare_data_from_raw.py` or by `run_all_from_raw.ps1` / `run_all_from_raw.sh`.

Expected generated files:

- `acs_harmonized_with_ids.csv.gz`;
- `oulad_all_registrations_horizons.csv.gz`;
- `south_german_with_ids.csv.gz`;
- `heart_disease_with_ids.csv.gz`.

Prepared tables are not tracked in the source repository. They can be reconstructed deterministically from the cited raw archives and the supplied manifests.
