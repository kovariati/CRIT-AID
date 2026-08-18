# Stable Discrimination Can Hide Reliability Failures in AI Decision Support under Distribution Shift and Changing Target Definitions

**CRIT-AID** is an executable empirical reliability-audit framework for AI decision support under distribution shift. It tests whether source-defined probability calibration, selection/abstention rules, and conformal uncertainty remain reliable when transported unchanged to target domains, when target definitions change, or when decision-relevant evidence is degraded. The **CRIT-AID protocol** is the framework's seven-stage execution procedure; neither term denotes a predictive model.

**Key finding: stable discrimination does not imply stable decision-support reliability.** Across four public tabular domains, discrimination could change little while probability calibration, transported operating points, or class-specific uncertainty changed materially. On identical ACS 2024 records, changing the income target definition left AUROC nearly unchanged while ECE differed by 0.083. Across 27 primary 90% conformal conditions, label-conditional calibration improved worst-class coverage in 18 but worsened it in 9 and usually enlarged prediction sets.

This repository is intentionally limited to the **analysis and reproducibility software layer**. Publication-production files and editorial history are kept outside the public research-code repository.

## Repository contents

```text
CRIT-AID/
├── scripts/              Data preparation, analyses, diagnostics, tables, figures, and audits
├── manifests/            Cohort, split, seed, provenance, harmonization, and analysis records
├── outputs/              Machine-readable results suitable for direct claim auditing
├── figures/              Current figures generated from machine-readable results
├── docs/                 Result registries and derived publication tables
├── data_raw/             User-supplied public source archives; not tracked by Git
├── prepared/             Locally generated analysis tables; not tracked by Git
└── .github/workflows/    Automated source-integrity audit
```

## What is stored directly in GitHub

The repository includes the analysis code, exact configurations, deterministic manifests, machine-readable summary and refit-level results, result registries, and generated figures. Most result files are small enough to remain in Git history and are therefore directly inspectable.

Three larger generated result files are distributed as an asset attached to the corresponding GitHub Release rather than committed to the repository history:

- `acs_risk_selection_curves.csv.gz`
- `oulad_risk_selection_curves.csv.gz`
- `acs_primary_target_predictions.csv.gz`

The release asset is optional for source-code inspection but required for a complete results audit. No external archive is required.

## Analysis specification

`manifests/analysis_matrix.csv` distinguishes the pre-specified primary and secondary analyses from post-hoc diagnostic analyses. Source data are separated into model-training, probability-calibration, selection/conformal-calibration, and ID-test partitions. Target data are not used to tune the model, probability mapping, selection threshold, or conformal quantile in the transport audit.

The four post-hoc diagnostic analyses are:

- identical-record ACS target-definition decomposition with prevalence/intercept alignment;
- marginal versus label-conditional conformal coverage-informativeness trade-off analysis;
- fixed cross-domain LightGBM model-family sensitivity analysis;
- descriptive cross-domain variance-component integration.

## Environment

The final environment is recorded in `requirements.txt`, `environment.yml`, and `manifests/environment.json`.

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass -Force
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux or macOS

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Source-only audit

This audit does not require public datasets or the large GitHub Release asset:

```bash
python scripts/audit_repository.py --mode source
```

## Recreate analyses from public source archives

Download the cited public archives and place them in `data_raw/` using the names in `manifests/raw_file_map.json`. Raw source archives are intentionally excluded from Git. Source-specific licenses and repository terms remain applicable; see `THIRD_PARTY_DATA.md`.

Windows:

```powershell
.\run_all_from_raw.ps1 -DataRoot .\data_raw
```

Linux or macOS:

```bash
CRIT_AID_DATA_ROOT="$PWD/data_raw" ./run_all_from_raw.sh
```

## Refit from prepared tables

```powershell
.\run_all_from_prepared.ps1
```

or

```bash
./run_all_from_prepared.sh
```

## Complete-results audit

After downloading the `CRIT_AID_large_results_v1.0.0.zip` asset from the GitHub Release, extract its `outputs/` contents into this repository's `outputs/` directory and run:

```bash
python scripts/audit_repository.py --mode results
```

## Data availability

OULAD, South German Credit, Heart Disease, and ACS PUMS are publicly available from their cited repositories. Raw source datasets are not redistributed here. `THIRD_PARTY_DATA.md` records source and licensing information, and `manifests/raw_input_manifest.csv` records expected source files and checksums.

## Citation

The canonical article title is:

**Stable Discrimination Can Hide Reliability Failures in AI Decision Support under Distribution Shift and Changing Target Definitions**

`CITATION.cff` provides GitHub's machine-readable citation metadata. The software object and the accompanying article remain distinct research objects. After journal publication, update only the preferred article citation with the final journal bibliographic record and DOI; the canonical title should remain unchanged.

## License

The source code is released under the MIT License. Third-party dataset licenses and repository terms remain separate.
