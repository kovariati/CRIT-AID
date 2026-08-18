from pathlib import Path
import os
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('CRIT_AID_ROOT', Path(__file__).resolve().parents[1]))
OUT = ROOT / 'outputs'


def pair_methods(df, keys, domain_name, condition_cols):
    use = df.copy()
    use['worst_class_coverage'] = use[['class0_coverage','class1_coverage']].min(axis=1)
    metrics = ['coverage','worst_class_coverage','avg_set_size','singleton_rate','ambiguous_rate','empty_rate']
    marg=use[use.conformal_method=='marginal'][keys+metrics].copy()
    lc=use[use.conformal_method=='label_conditional'][keys+metrics].copy()
    marg=marg.rename(columns={m:f'marginal_{m}' for m in metrics})
    lc=lc.rename(columns={m:f'label_conditional_{m}' for m in metrics})
    out=marg.merge(lc,on=keys,how='inner',validate='one_to_one')
    out['domain_family'] = domain_name
    out['condition_id'] = out[condition_cols].astype(str).agg('|'.join, axis=1)
    for m in metrics:
        out[f'delta_{m}'] = out[f'label_conditional_{m}'] - out[f'marginal_{m}']
    return out


def main():
    pieces=[]

    # ACS primary: logistic regression, Platt, survey-weighted, OOD target, alpha=.10.
    a=pd.read_csv(OUT/'acs_conformal_complete.csv.gz')
    a=a[(a.model=='logistic_regression') & (a.mapping=='platt') & (a.weighting=='survey_weighted') &
        (a.domain=='ood_test') & np.isclose(a.alpha,0.10)]
    pieces.append(pair_methods(a, ['repeat','target_definition'], 'ACSIncome', ['target_definition']))

    # OULAD primary: exact pair analyses, score-free representation, Platt, OOD, alpha=.10.
    o=pd.read_csv(OUT/'oulad_conformal_complete.csv.gz')
    o=o[(o.analysis_id!='pooled_fixed_effects') & (o.representation=='score_free') &
        (o.model=='logistic_regression') & (o.mapping=='platt') & (o.domain=='ood_test') & np.isclose(o.alpha,0.10)]
    pieces.append(pair_methods(o, ['repeat','analysis_id','horizon_day'], 'OULAD', ['analysis_id','horizon_day']))

    # South German Credit primary stress comparison: clean and targeted top-three feature deletion.
    s=pd.read_csv(OUT/'south_conformal_complete.csv.gz')
    s=s[(s.model=='logistic_regression') & (s.mapping=='platt') & s.scenario.isin(['clean','targeted_top3']) & np.isclose(s.alpha,0.10)]
    pieces.append(pair_methods(s, ['repeat','scenario'], 'SouthGermanCredit', ['scenario']))

    # Heart primary: leave-one-site-out OOD, no missing indicators, Platt, alpha=.10.
    h=pd.read_csv(OUT/'heart_conformal_complete.csv.gz')
    h=h[(h.model=='logistic_regression') & (h.mapping=='platt') & (~h.missing_indicators.astype(bool)) &
        (h.domain=='ood_test') & np.isclose(h.alpha,0.10)]
    pieces.append(pair_methods(h, ['repeat','heldout_site'], 'HeartDisease', ['heldout_site']))

    d=pd.concat(pieces, ignore_index=True, sort=False)
    d['delta_worst_class_coverage']=d['delta_worst_class_coverage'].astype(float)
    d.to_csv(OUT/'conformal_tradeoff_complete.csv', index=False)

    condition=(d.groupby(['domain_family','condition_id'],as_index=False)
                 .agg(n_repeats=('repeat','size'),
                      marginal_worst_class_coverage=('marginal_worst_class_coverage','mean'),
                      label_conditional_worst_class_coverage=('label_conditional_worst_class_coverage','mean'),
                      delta_worst_class_coverage=('delta_worst_class_coverage','mean'),
                      marginal_avg_set_size=('marginal_avg_set_size','mean'),
                      label_conditional_avg_set_size=('label_conditional_avg_set_size','mean'),
                      delta_avg_set_size=('delta_avg_set_size','mean'),
                      marginal_singleton_rate=('marginal_singleton_rate','mean'),
                      label_conditional_singleton_rate=('label_conditional_singleton_rate','mean'),
                      delta_singleton_rate=('delta_singleton_rate','mean')))
    tol=1e-12
    condition['worst_class_coverage_direction']=np.where(condition.delta_worst_class_coverage>tol,'improved',np.where(condition.delta_worst_class_coverage<-tol,'worsened','unchanged'))
    condition.to_csv(OUT/'conformal_tradeoff_condition_summary.csv',index=False)

    repdir=np.where(d.delta_worst_class_coverage>tol,'improved',np.where(d.delta_worst_class_coverage<-tol,'worsened','unchanged'))
    overview=pd.DataFrame([{
        'n_primary_conditions':len(condition),
        'conditions_improved':int((condition.worst_class_coverage_direction=='improved').sum()),
        'conditions_worsened':int((condition.worst_class_coverage_direction=='worsened').sum()),
        'conditions_unchanged':int((condition.worst_class_coverage_direction=='unchanged').sum()),
        'n_repeated_comparisons':len(d),
        'repeats_improved':int((repdir=='improved').sum()),
        'repeats_worsened':int((repdir=='worsened').sum()),
        'repeats_unchanged':int((repdir=='unchanged').sum()),
        'repeats_larger_set':int((d.delta_avg_set_size>tol).sum()),
        'repeats_smaller_set':int((d.delta_avg_set_size<-tol).sum()),
        'repeats_same_set':int((abs(d.delta_avg_set_size)<=tol).sum()),
        'fraction_repeats_larger_set':float((d.delta_avg_set_size>tol).mean()),
        'mean_delta_worst_class_coverage':float(d.delta_worst_class_coverage.mean()),
        'mean_delta_avg_set_size':float(d.delta_avg_set_size.mean()),
        'mean_delta_singleton_rate':float(d.delta_singleton_rate.mean())
    }])
    overview.to_csv(OUT/'conformal_tradeoff_overall_summary.csv',index=False)
    print(overview.to_string(index=False))
    print('\nCondition extremes:')
    print(condition.sort_values('delta_worst_class_coverage')[['domain_family','condition_id','delta_worst_class_coverage','delta_avg_set_size','delta_singleton_rate']].to_string(index=False))

if __name__=='__main__':
    main()
