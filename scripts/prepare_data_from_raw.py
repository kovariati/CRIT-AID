from __future__ import annotations

"""Create CRIT-AID prepared tables from the cited public raw archives.

The script expects the raw ZIP archives in CRIT_AID_DATA_ROOT (default: data_raw/
inside the release). Filenames can be changed in manifests/raw_file_map.json.
It does not download data and does not redistribute source archives.
"""

import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import sha256_file

ROOT = Path(os.environ.get("CRIT_AID_ROOT", Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(os.environ.get("CRIT_AID_DATA_ROOT", ROOT / "data_raw"))
PREP = ROOT / "prepared"
MAN = ROOT / "manifests"
OUT = ROOT / "outputs"
for d in (PREP, MAN, OUT):
    d.mkdir(parents=True, exist_ok=True)

MAP_PATH = MAN / "raw_file_map.json"
DEFAULT_MAP = {
    "oulad": "open+university+learning+analytics+dataset.zip",
    "south_german_credit": "south+german+credit.zip",
    "heart_disease": "heart+disease.zip",
    "acs_2018_CA": "csv_pca_2018.zip",
    "acs_2018_FL": "csv_pfl 2018.zip",
    "acs_2018_NY": "csv_pny 2018.zip",
    "acs_2018_TX": "csv_ptx 2018.zip",
    "acs_2024_CA": "csv_pca.zip",
    "acs_2024_FL": "csv_pfl.zip",
    "acs_2024_NY": "csv_pny.zip",
    "acs_2024_TX": "csv_ptx.zip",
}
REL_MAP_2024_TO_2018 = {
    20: 0, 21: 1, 22: 13, 23: 1, 24: 13, 25: 2, 26: 3, 27: 4,
    28: 5, 29: 6, 30: 7, 31: 8, 32: 9, 33: 10, 34: 12, 35: 14,
    36: 15, 37: 16, 38: 17,
}
HORIZONS = [14, 28, 42, 56]
OULAD_KEYS = ["code_module", "code_presentation", "id_student"]


def load_file_map() -> dict[str, str]:
    if not MAP_PATH.exists():
        MAP_PATH.write_text(json.dumps(DEFAULT_MAP, indent=2), encoding="utf-8")
    mapping = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    missing = [k for k in DEFAULT_MAP if k not in mapping]
    if missing:
        raise KeyError(f"Missing keys in {MAP_PATH}: {missing}")
    return mapping


def path_for(mapping: dict[str, str], key: str) -> Path:
    p = DATA_ROOT / mapping[key]
    if not p.exists():
        raise FileNotFoundError(f"Missing raw archive for {key}: {p}")
    return p


def write_raw_manifest(mapping: dict[str, str]) -> None:
    rows = []
    for key, name in mapping.items():
        p = DATA_ROOT / name
        if not p.exists():
            continue
        with zipfile.ZipFile(p) as zf:
            bad = zf.testzip()
            members = [x.filename for x in zf.infolist() if not x.is_dir()]
        rows.append({
            "source_key": key,
            "filename": p.name,
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "zip_valid": bad is None,
            "members": "|".join(members),
        })
    pd.DataFrame(rows).to_csv(MAN / "raw_input_manifest.csv", index=False)


def harmonize_relationship(year: int, values: pd.Series) -> pd.Series:
    if year == 2018:
        return pd.to_numeric(values, errors="coerce").astype("Int64")
    return pd.to_numeric(values, errors="coerce").map(REL_MAP_2024_TO_2018).astype("Int64")


def prepare_acs(mapping: dict[str, str]) -> None:
    keep = [
        "SERIALNO", "SPORDER", "AGEP", "COW", "SCHL", "MAR", "OCCP",
        "POBP", "WKHP", "SEX", "RAC1P", "PINCP", "ADJINC", "PWGTP",
    ]
    frames = []
    inventory = []
    for year in (2018, 2024):
        rel_col = "RELP" if year == 2018 else "RELSHIPP"
        for state in ("CA", "FL", "NY", "TX"):
            p = path_for(mapping, f"acs_{year}_{state}")
            with zipfile.ZipFile(p) as zf:
                names = [n for n in zf.namelist() if Path(n).name.lower().startswith("psam_") and n.lower().endswith(".csv")]
                if len(names) != 1:
                    raise RuntimeError(f"Expected one person CSV in {p}, found {names}")
                raw_rows = 0
                eligible_rows = 0
                parts = []
                with zf.open(names[0]) as fh:
                    for chunk in pd.read_csv(fh, usecols=keep + [rel_col], chunksize=200_000, low_memory=False):
                        raw_rows += len(chunk)
                        for c in ["AGEP", "WKHP", "PINCP", "PWGTP", "ADJINC", "SPORDER"]:
                            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
                        mask = (chunk.AGEP > 16) & (chunk.PINCP > 100) & (chunk.WKHP > 0) & (chunk.PWGTP >= 1)
                        d = chunk.loc[mask].copy()
                        eligible_rows += len(d)
                        d["year"] = year
                        d["state"] = state
                        d["REL_ORIGINAL"] = pd.to_numeric(d[rel_col], errors="coerce").astype("Int64")
                        d["REL_HARM"] = harmonize_relationship(year, d[rel_col])
                        d["PINCP_ADJ_YEAR_DOLLARS"] = d.PINCP * d.ADJINC / 1_000_000.0
                        d["target_unadjusted_50000"] = (d.PINCP > 50_000).astype("int8")
                        d["target_survey_year_adjusted_50000"] = (d.PINCP_ADJ_YEAR_DOLLARS > 50_000).astype("int8")
                        serial = d.SERIALNO.astype(str).str.replace('"', "", regex=False)
                        sporder = d.SPORDER.astype("Int64").astype(str)
                        d["record_id"] = serial + "-" + sporder
                        d["record_key"] = str(year) + "-" + state + "-" + d.record_id
                        parts.append(d)
                out = pd.concat(parts, ignore_index=True)
                frames.append(out[[
                    "record_key", "record_id", "year", "state", "AGEP", "COW", "SCHL", "MAR",
                    "OCCP", "POBP", "WKHP", "SEX", "RAC1P", "REL_HARM", "REL_ORIGINAL",
                    "PINCP", "ADJINC", "PINCP_ADJ_YEAR_DOLLARS", "PWGTP",
                    "target_unadjusted_50000", "target_survey_year_adjusted_50000",
                ]])
                inventory.append({"year": year, "state": state, "source_file": p.name,
                                  "raw_rows": raw_rows, "eligible_rows": eligible_rows})
                print(f"ACS {year} {state}: {eligible_rows:,}/{raw_rows:,}", flush=True)
    acs = pd.concat(frames, ignore_index=True)
    acs.to_csv(PREP / "acs_harmonized_with_ids.csv.gz", index=False, compression="gzip")
    pd.DataFrame(inventory).to_csv(MAN / "acs_source_inventory.csv", index=False)


def load_oulad_static(p: Path):
    with zipfile.ZipFile(p) as zf:
        def read(name: str, **kwargs):
            with zf.open(name) as fh:
                return pd.read_csv(fh, **kwargs)
        return (
            read("studentInfo.csv"),
            read("studentRegistration.csv", na_values=["?", ""]),
            read("vle.csv", usecols=["code_module", "code_presentation", "id_site", "activity_type"]),
            read("assessments.csv", na_values=["?", ""]),
            read("studentAssessment.csv", na_values=["?", ""]),
        )


def prepare_oulad(mapping: dict[str, str]) -> None:
    source = path_for(mapping, "oulad")
    info, reg, vle_map, assessments, student_assessment = load_oulad_static(source)
    dtypes = {"code_module": "string", "code_presentation": "string", "id_student": "int32",
              "id_site": "int32", "date": "int16", "sum_click": "int32"}
    retained = []
    with zipfile.ZipFile(source) as zf, zf.open("studentVle.csv") as fh:
        for chunk in pd.read_csv(fh, dtype=dtypes, chunksize=500_000):
            q = chunk.loc[chunk.date <= max(HORIZONS)].copy()
            if len(q):
                retained.append(q)
    raw = pd.concat(retained, ignore_index=True)
    vle_map = vle_map.astype({"code_module": "string", "code_presentation": "string"})
    raw = raw.merge(vle_map, on=["code_module", "code_presentation", "id_site"], how="left", validate="many_to_one")
    if raw.activity_type.isna().any():
        raise RuntimeError("Unmapped OULAD activity type")
    activity_types = sorted(raw.activity_type.unique())
    vle_parts = []
    for horizon in HORIZONS:
        d = raw.loc[raw.date <= horizon, OULAD_KEYS + ["date", "sum_click", "activity_type"]]
        base = d.groupby(OULAD_KEYS, observed=True).agg(
            clicks_total=("sum_click", "sum"), active_days=("date", "nunique"),
            vle_record_count=("sum_click", "size"), first_activity_day=("date", "min"),
            last_activity_day=("date", "max"),
        ).reset_index()
        base["days_since_last_activity"] = horizon - base.last_activity_day
        by_type = d.groupby(OULAD_KEYS + ["activity_type"], observed=True).sum_click.sum().unstack(fill_value=0)
        by_type = by_type.reindex(columns=activity_types, fill_value=0)
        by_type.columns = [f"clicks_{x}" for x in by_type.columns]
        by_type = by_type.reset_index()
        base = base.merge(by_type, on=OULAD_KEYS, validate="one_to_one")
        base["horizon_day"] = horizon
        vle_parts.append(base)
    vle = pd.concat(vle_parts, ignore_index=True)

    a = assessments.copy()
    a["date"] = pd.to_numeric(a.date, errors="coerce")
    a = a.loc[a.date.notna()].copy()
    a["date"] = a.date.astype(int)
    a["weight"] = pd.to_numeric(a.weight, errors="coerce").fillna(0.0)
    sa = student_assessment.merge(
        a[["code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"]],
        on="id_assessment", how="left", validate="many_to_one")
    sa["date_submitted"] = pd.to_numeric(sa.date_submitted, errors="coerce")
    sa["score"] = pd.to_numeric(sa.score, errors="coerce")

    cohort = info.merge(reg, on=OULAD_KEYS, validate="one_to_one")
    cohort["target_unsuccessful"] = cohort.final_result.isin(["Withdrawn", "Fail"]).astype("int8")
    full_parts = []
    for horizon in HORIZONS:
        c = cohort.copy()
        c["horizon_day"] = horizon
        c["registered_by_horizon"] = (c.date_registration.isna() | (c.date_registration <= horizon)).astype("int8")
        c["withdrawn_by_horizon"] = (c.date_unregistration.notna() & (c.date_unregistration <= horizon)).astype("int8")
        due = a.loc[a.date <= horizon].groupby(["code_module", "code_presentation"], as_index=False).agg(
            assessments_due=("id_assessment", "nunique"), assessment_weight_due=("weight", "sum"))
        sh = sa.loc[(sa.date <= horizon) & (sa.date_submitted <= horizon)].copy()
        sh["late"] = (sh.date_submitted > sh.date).astype(int)
        sh["weighted_score_component"] = sh.score * sh.weight / 100.0
        actual = sh.groupby(OULAD_KEYS, as_index=False).agg(
            assessments_submitted=("id_assessment", "nunique"), mean_score=("score", "mean"),
            weighted_score_earned=("weighted_score_component", "sum"), submitted_weight=("weight", "sum"),
            late_submissions=("late", "sum"), banked_submissions=("is_banked", "sum"))
        feat = c.merge(vle.loc[vle.horizon_day == horizon], on=OULAD_KEYS + ["horizon_day"], how="left", validate="one_to_one")
        feat = feat.merge(due, on=["code_module", "code_presentation"], how="left", validate="many_to_one")
        feat = feat.merge(actual, on=OULAD_KEYS, how="left", validate="one_to_one")
        count_cols = [x for x in feat.columns if x.startswith("clicks_")] + [
            "active_days", "vle_record_count", "assessments_due", "assessment_weight_due",
            "assessments_submitted", "weighted_score_earned", "submitted_weight", "late_submissions",
            "banked_submissions"]
        for col in count_cols:
            feat[col] = feat[col].fillna(0)
        feat["submission_ratio"] = np.where(feat.assessments_due > 0, feat.assessments_submitted / feat.assessments_due, np.nan)
        feat["late_submission_ratio"] = np.where(feat.assessments_submitted > 0, feat.late_submissions / feat.assessments_submitted, np.nan)
        feat["weighted_score_fraction_due"] = np.where(feat.assessment_weight_due > 0, feat.weighted_score_earned / feat.assessment_weight_due, np.nan)
        feat["no_vle_activity"] = (feat.active_days == 0).astype("int8")
        feat["no_submitted_assessment"] = (feat.assessments_submitted == 0).astype("int8")
        feat["row_key"] = (feat.code_module.astype(str) + "-" + feat.code_presentation.astype(str) + "-" +
                           feat.id_student.astype(str) + "-d" + str(horizon))
        full_parts.append(feat)
    full = pd.concat(full_parts, ignore_index=True)
    full.to_csv(PREP / "oulad_all_registrations_horizons.csv.gz", index=False, compression="gzip")
    print(f"OULAD prepared rows: {len(full):,}", flush=True)


def prepare_external(mapping: dict[str, str]) -> None:
    south_p = path_for(mapping, "south_german_credit")
    with zipfile.ZipFile(south_p) as zf:
        d = pd.read_csv(zf.open("SouthGermanCredit.asc"), sep=r"\s+")
    d.insert(0, "row_id", np.arange(len(d), dtype=int))
    d.to_csv(PREP / "south_german_with_ids.csv.gz", index=False, compression="gzip")

    heart_p = path_for(mapping, "heart_disease")
    files = {"Cleveland": "processed.cleveland.data", "Hungary": "processed.hungarian.data",
             "Switzerland": "processed.switzerland.data", "VA Long Beach": "processed.va.data"}
    cols = ["age", "sex", "chest_pain_type", "resting_bp", "cholesterol", "fasting_blood_sugar",
            "resting_ecg", "max_heart_rate", "exercise_angina", "oldpeak", "slope", "major_vessels",
            "thal", "disease_severity"]
    frames, provenance = [], []
    with zipfile.ZipFile(heart_p) as zf:
        for site, filename in files.items():
            x = pd.read_csv(io.StringIO(zf.read(filename).decode("latin1")), names=cols, na_values="?")
            x.insert(0, "site_row_id", np.arange(len(x), dtype=int))
            x.insert(0, "site", site)
            x["record_key"] = site.replace(" ", "_") + "-" + x.site_row_id.astype(str)
            x["target_disease_present"] = (pd.to_numeric(x.disease_severity, errors="coerce") > 0).astype("int8")
            frames.append(x)
            provenance.append({"site": site, "file": filename, "n": len(x),
                               "positive_n": int(x.target_disease_present.sum()),
                               "negative_n": int((1 - x.target_disease_present).sum()),
                               "prevalence": float(x.target_disease_present.mean()),
                               "missing_cells": int(x[cols[:-1]].isna().sum().sum())})
    pd.concat(frames, ignore_index=True).to_csv(PREP / "heart_disease_with_ids.csv.gz", index=False, compression="gzip")
    pd.DataFrame(provenance).to_csv(MAN / "heart_provenance.csv", index=False)


def main() -> None:
    mapping = load_file_map()
    write_raw_manifest(mapping)
    prepare_acs(mapping)
    prepare_oulad(mapping)
    prepare_external(mapping)
    print("Prepared-data generation complete.", flush=True)


if __name__ == "__main__":
    main()
