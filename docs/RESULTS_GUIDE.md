# Results guide

The result files are organized by scientific function rather than review history.

## Primary domain results

- `acs_*`: ACS target-definition sensitivity, calibration, selection, conformal, subgroup, and paired analyses.
- `oulad_*`: OULAD exact module-period transfer, estimand sensitivity, selection, conformal, and two-stage bootstrap analyses.
- `south_*`: South German Credit evidence-degradation analyses and targeted-versus-random controls.
- `heart_*`: multi-site Heart Disease transfer, calibration, selection, and class-specific conformal results.

## Cross-domain diagnostics

- `acs_target_label_change_summary.csv`
- `acs_two_factor_metric_decomposition.csv`
- `acs_prevalence_alignment_summary.csv`
- `conformal_tradeoff_*`
- `model_family_sensitivity_*`
- `model_family_delta_effects.csv`
- `cross_domain_shift_deltas.csv`
- `cross_domain_variance_components.csv`

These analyses are explicitly marked as post-hoc diagnostics in `manifests/analysis_matrix.csv`. Their scientific status is unchanged by the public-file renaming.

## Large GitHub Release asset

The following generated files are intentionally excluded from Git history because of size and distributed as a GitHub Release asset:

- `acs_risk_selection_curves.csv.gz`
- `oulad_risk_selection_curves.csv.gz`
- `acs_primary_target_predictions.csv.gz`

All other current result files are committed directly to the repository.
