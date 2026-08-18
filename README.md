# Stable Discrimination Can Hide Reliability Failures in AI Decision Support under Distribution Shift and Changing Target Definitions

CRIT-AID is an executable empirical reliability-audit framework for AI decision support under distribution shift. The framework comprises a seven-stage audit protocol, stress-test specifications, a common reliability-metric interface, interpretation rules, deterministic provenance records, and machine-readable reproducibility artifacts.

The CRIT-AID protocol is the framework's mandatory seven-stage execution procedure. It tests whether a probabilistic AI decision-support pipeline preserves specified reliability properties when a source-defined estimand, probability mapping, operating rule, or evidence structure is transported to a target domain or controlled stress condition. A conforming execution prohibits target-test tuning, preserves disjoint fitting/calibration/test roles, and jointly reports discrimination, probability quality, selective behavior, conformal validity, and uncertainty informativeness. CRIT-AID is therefore an audit framework and protocol, not a predictive model or a new learning algorithm.

In this project, a **reliability failure** refers to loss of preservation of a stated statistical reliability property under transport or stress. It does not by itself imply an application-level safety failure, clinical harm, governance failure, or pass/fail judgment; such conclusions require domain-specific tolerances and downstream evidence.

## Key finding

**Stable discrimination does not imply stable decision-support reliability.**

Across four public tabular domains, stable discrimination did not imply stable probability quality, selective operating points, or conformal uncertainty. On identical ACS 2024 records, changing the income target definition left AUROC nearly unchanged while ECE differed by 0.083; prevalence-intercept alignment reduced this difference to approximately -0.004. This shows that target semantics can alter probability reliability without materially changing ranking.

Across 27 primary 90% conformal conditions, label-conditional calibration improved worst-class coverage in 18 conditions but worsened it in 9 and usually increased prediction-set size. The result therefore represents a coverage–informativeness trade-off rather than a universal reliability improvement.

A fixed cross-domain LightGBM sensitivity analysis changed absolute discrimination and other performance quantities without eliminating the mismatch among reliability dimensions.

The general implication is deliberately narrow and testable:

> Stable discrimination is insufficient evidence that probabilities, transported operating rules, or uncertainty outputs remain reliable when deployment conditions or target definitions change.

The relevant object of a deployment reliability audit is therefore not discrimination alone, but the **transportability of probability mappings, operating rules, class-specific validity, and uncertainty informativeness**.

## Repository scope

This repository is intentionally limited to the analysis and reproducibility software layer of the study. Manuscript-production files, reviewer correspondence, revision history, and editorial materials are kept outside the public research-code repository.

The repository contains the code and machine-readable artifacts required to reproduce and audit the empirical analyses reported in the article.

## Repository contents

```text
CRIT-AID/
├── scripts/              Data preparation, analyses, diagnostics, tables, figures, and audits
├── manifests/            Cohort, split, seed, provenance, harmonization, and analysis records
├── outputs/              Machine-readable results suitable for direct claim auditing
├── figures/              Figures generated from machine-readable results
├── docs/                 Result registries and derived machine-readable summary tables
├── data_raw/             User-supplied public source archives; not tracked by Git
├── prepared/             Locally generated analysis tables; not tracked by Git
└── .github/workflows/    Automated source-integrity audit
