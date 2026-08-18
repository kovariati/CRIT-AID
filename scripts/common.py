from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

EPS = 1e-8


def weighted_mean(x, w=None) -> float:
    x = np.asarray(x, dtype=float)
    if w is None:
        return float(np.mean(x))
    w = np.asarray(w, dtype=float)
    return float(np.average(x, weights=w))


def weighted_quantile(values, q, weights=None) -> float:
    values = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.quantile(values, q, method='linear'))
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values, kind='mergesort')
    v = values[order]
    w = weights[order]
    cumulative = np.cumsum(w) - 0.5 * w
    cumulative /= w.sum()
    return float(np.interp(q, cumulative, v))


def expected_calibration_error(y, p, w=None, n_bins=15):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if w is None:
        w = np.ones(len(y), dtype=float)
    else:
        w = np.asarray(w, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    total = w.sum()
    ece = 0.0
    rows = []
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        wb = w[m]
        mass = wb.sum()
        obs = float(np.average(y[m], weights=wb))
        pred = float(np.average(p[m], weights=wb))
        gap = abs(obs - pred)
        ece += mass / total * gap
        rows.append({
            'bin': b,
            'left': edges[b],
            'right': edges[b+1],
            'n': int(m.sum()),
            'weight': float(mass),
            'observed_rate': obs,
            'mean_probability': pred,
            'absolute_gap': gap,
        })
    return float(ece), pd.DataFrame(rows)


def calibration_intercept_slope(y, p, w=None):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e8, solver='lbfgs', max_iter=3000, tol=1e-9)
    try:
        model.fit(z, np.asarray(y, dtype=int), sample_weight=w)
        return float(model.intercept_[0]), float(model.coef_[0, 0]), int(model.n_iter_.max())
    except Exception:
        return np.nan, np.nan, 0


def safe_auroc(y, p, w=None):
    try:
        return float(roc_auc_score(y, p, sample_weight=w))
    except Exception:
        return np.nan


def safe_ap(y, p, w=None):
    try:
        return float(average_precision_score(y, p, sample_weight=w))
    except Exception:
        return np.nan


def risk_selection_curve(y, p, w=None):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    err = (pred != y).astype(float)
    conf = np.maximum(p, 1 - p)
    if w is None:
        w = np.ones(len(y), dtype=float)
    else:
        w = np.asarray(w, dtype=float)
    order = np.argsort(-conf, kind='mergesort')
    conf_o = conf[order]
    err_o = err[order]
    w_o = w[order]
    cumw = np.cumsum(w_o)
    selection_rate = cumw / cumw[-1]
    risk = np.cumsum(w_o * err_o) / cumw
    # Keep one point per distinct confidence after all tied observations are included.
    last_of_tie = np.r_[conf_o[1:] != conf_o[:-1], True]
    sr = selection_rate[last_of_tie]
    rr = risk[last_of_tie]
    aurc = float(np.trapezoid(rr, sr))
    # Correct-first lower bound under the same weighted error burden.
    oo = np.argsort(err, kind='mergesort')
    ew = err[oo]
    ww = w[oo]
    oc = np.cumsum(ww) / ww.sum()
    orisk = np.cumsum(ww * ew) / np.cumsum(ww)
    oracle_aurc = float(np.trapezoid(orisk, oc))
    return pd.DataFrame({'selection_rate': sr, 'selective_risk': rr}), aurc, aurc - oracle_aurc


def metrics(y, p, w=None, n_bins=15):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    if w is not None:
        w = np.asarray(w, dtype=float)
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1 - p)
    wrong = pred != y
    ece, _ = expected_calibration_error(y, p, w=w, n_bins=n_bins)
    intercept, slope, cal_iter = calibration_intercept_slope(y, p, w=w)
    _, aurc, eaurc = risk_selection_curve(y, p, w=w)
    prevalence = weighted_mean(y, w)
    ap = safe_ap(y, p, w)
    pr_skill = (ap - prevalence) / (1 - prevalence) if np.isfinite(ap) and prevalence < 1 else np.nan

    def wm(mask):
        return weighted_mean(np.asarray(mask, dtype=float), w)

    return {
        'n': int(len(y)),
        'weight_sum': float(np.sum(w)) if w is not None else float(len(y)),
        'prevalence': prevalence,
        'auroc': safe_auroc(y, p, w),
        'average_precision': ap,
        'pr_skill': pr_skill,
        'brier': float(brier_score_loss(y, p, sample_weight=w)),
        'log_loss': float(log_loss(y, p, labels=[0, 1], sample_weight=w)),
        'ece': ece,
        'calibration_intercept': intercept,
        'calibration_slope': slope,
        'calibration_fit_iterations': cal_iter,
        'hcep_080': wm(wrong & (conf >= 0.80)),
        'hcep_090': wm(wrong & (conf >= 0.90)),
        'hcep_095': wm(wrong & (conf >= 0.95)),
        'aurc': aurc,
        'excess_aurc': eaurc,
    }


class IdentityCalibrator:
    name = 'raw'
    def fit(self, p, y, sample_weight=None):
        self.n_iter_ = 0
        return self
    def predict(self, p):
        return np.asarray(p, dtype=float)


class PlattCalibrator:
    name = 'platt'
    def __init__(self):
        self.model = LogisticRegression(C=1e8, solver='lbfgs', max_iter=3000, tol=1e-9)
    def fit(self, p, y, sample_weight=None):
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        z = np.log(p / (1 - p)).reshape(-1, 1)
        self.model.fit(z, np.asarray(y, dtype=int), sample_weight=sample_weight)
        self.n_iter_ = int(self.model.n_iter_.max())
        return self
    def predict(self, p):
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        z = np.log(p / (1 - p)).reshape(-1, 1)
        return self.model.predict_proba(z)[:, 1]


class IsotonicCalibrator:
    name = 'isotonic'
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    def fit(self, p, y, sample_weight=None):
        self.model.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=int), sample_weight=sample_weight)
        self.n_iter_ = 0
        return self
    def predict(self, p):
        return np.asarray(self.model.predict(np.asarray(p, dtype=float)), dtype=float)


def fit_calibrators(y_cal, p_cal, w_cal=None):
    result = {}
    for cal in [IdentityCalibrator(), PlattCalibrator(), IsotonicCalibrator()]:
        try:
            result[cal.name] = cal.fit(p_cal, y_cal, sample_weight=w_cal)
        except Exception:
            result[cal.name] = None
    return result


def selection_threshold(confidence_cal, desired_rate, weights=None):
    return weighted_quantile(confidence_cal, 1 - desired_rate, weights)


def selective_metrics(y, p, threshold, w=None):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    conf = np.maximum(p, 1 - p)
    selected = conf >= float(threshold)
    weights = np.ones(len(y), dtype=float) if w is None else np.asarray(w, dtype=float)
    rate = float(np.average(selected.astype(float), weights=weights))
    if not selected.any():
        return {'selection_rate': rate, 'selected_n': 0, 'selective_risk': np.nan, 'selective_accuracy': np.nan}
    pred = (p >= 0.5).astype(int)
    risk = float(np.average((pred[selected] != y[selected]).astype(float), weights=weights[selected]))
    return {'selection_rate': rate, 'selected_n': int(selected.sum()), 'selective_risk': risk, 'selective_accuracy': 1-risk}


def conformal_rank(n: int, alpha: float) -> int:
    return min(n, math.ceil((n + 1) * (1 - alpha)))


def marginal_qhat(y_cal, p_cal, alpha=0.10):
    y_cal = np.asarray(y_cal, dtype=int)
    p_cal = np.asarray(p_cal, dtype=float)
    scores = 1 - np.where(y_cal == 1, p_cal, 1 - p_cal)
    n = len(scores)
    k = conformal_rank(n, alpha)
    return float(np.sort(scores, kind='mergesort')[k-1]), k, n


def label_conditional_qhat(y_cal, p_cal, alpha=0.10):
    y_cal = np.asarray(y_cal, dtype=int)
    p_cal = np.asarray(p_cal, dtype=float)
    out = {}
    for c in [0, 1]:
        m = y_cal == c
        score = 1 - (1 - p_cal[m] if c == 0 else p_cal[m])
        n = int(m.sum())
        if n == 0:
            out[c] = {'qhat': np.nan, 'k': 0, 'n': 0, 'rank_fraction': np.nan}
        else:
            k = conformal_rank(n, alpha)
            q = float(np.sort(score, kind='mergesort')[k-1])
            out[c] = {'qhat': q, 'k': k, 'n': n, 'rank_fraction': k/(n+1)}
    return out


def conformal_metrics(y, p, qhat, w=None, method='marginal'):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    weights = np.ones(len(y), dtype=float) if w is None else np.asarray(w, dtype=float)
    if method == 'marginal':
        q0 = q1 = float(qhat)
    elif method == 'label_conditional':
        q0 = float(qhat[0]['qhat'])
        q1 = float(qhat[1]['qhat'])
    else:
        raise ValueError(method)
    include0 = p <= q0
    include1 = (1 - p) <= q1
    size = include0.astype(int) + include1.astype(int)
    covered = np.where(y == 1, include1, include0)
    singleton = size == 1
    singleton_pred = include1.astype(int)

    def avg(a, mask=None):
        if mask is None:
            mask = np.ones(len(y), dtype=bool)
        if not np.any(mask):
            return np.nan
        return float(np.average(np.asarray(a, dtype=float)[mask], weights=weights[mask]))

    out = {
        'conformal_method': method,
        'coverage': avg(covered),
        'avg_set_size': avg(size),
        'singleton_rate': avg(singleton),
        'ambiguous_rate': avg(size == 2),
        'empty_rate': avg(size == 0),
        'singleton_accuracy': avg(singleton_pred == y, singleton),
    }
    for c in [0, 1]:
        m = y == c
        sm = m & singleton
        out[f'class{c}_n'] = int(m.sum())
        out[f'class{c}_coverage'] = avg(covered, m)
        out[f'class{c}_avg_set_size'] = avg(size, m)
        out[f'class{c}_singleton_rate'] = avg(singleton, m)
        out[f'class{c}_singleton_accuracy'] = avg(singleton_pred == y, sm)
        out[f'class{c}_empty_rate'] = avg(size == 0, m)
    return out


def clopper_pearson(successes: int, n: int, alpha=0.05):
    if n <= 0:
        return np.nan, np.nan
    low = 0.0 if successes == 0 else float(beta.ppf(alpha/2, successes, n-successes+1))
    high = 1.0 if successes == n else float(beta.ppf(1-alpha/2, successes+1, n-successes))
    return low, high


def empirical_summary(df: pd.DataFrame, groups: Sequence[str], metrics_cols: Iterable[str]):
    rows = []
    for keys, g in df.groupby(list(groups), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(groups, keys))
        for metric in metrics_cols:
            vals = pd.to_numeric(g[metric], errors='coerce').dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            rows.append({
                **base,
                'metric': metric,
                'n_repeats': int(len(vals)),
                'mean': float(vals.mean()),
                'sd': float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                'min': float(vals.min()),
                'q25': float(np.quantile(vals, 0.25)),
                'median': float(np.quantile(vals, 0.50)),
                'q75': float(np.quantile(vals, 0.75)),
                'max': float(vals.max()),
            })
    return pd.DataFrame(rows)


def paired_summary(df: pd.DataFrame, groups: Sequence[str], value_col='delta'):
    rows = []
    for keys, g in df.groupby(list(groups), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        v = pd.to_numeric(g[value_col], errors='coerce').dropna().to_numpy(float)
        if len(v) == 0:
            continue
        rows.append({
            **dict(zip(groups, keys)),
            'n': len(v),
            'mean': v.mean(),
            'sd': v.std(ddof=1) if len(v) > 1 else 0.0,
            'min': v.min(),
            'median': np.median(v),
            'max': v.max(),
        })
    return pd.DataFrame(rows)


def save_environment(path: Path):
    import numpy, pandas, scipy, sklearn
    try:
        import lightgbm
        lgb = lightgbm.__version__
    except Exception:
        lgb = None
    data = {
        'python': sys.version,
        'platform': platform.platform(),
        'numpy': numpy.__version__,
        'pandas': pandas.__version__,
        'scipy': scipy.__version__,
        'scikit_learn': sklearn.__version__,
        'lightgbm': lgb,
    }
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')

 
