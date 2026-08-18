from pathlib import Path
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

ROOT=Path(os.environ.get('CRIT_AID_ROOT',Path(__file__).resolve().parents[1])); OUT=ROOT/'outputs'
METRICS=['auroc','ece','brier']

def prepare():
    d=pd.read_csv(OUT/'model_family_sensitivity_complete.csv.gz')
    idx=['domain_family','condition','repeat','seed','model']
    w=d.pivot_table(index=idx,columns='domain',values=METRICS,aggfunc='first').reset_index()
    w.columns=[c[0] if isinstance(c,tuple) and c[1]=='' else (f'{c[0]}_{c[1]}' if isinstance(c,tuple) else c) for c in w.columns]
    for m in METRICS:
        w[f'delta_{m}']=w[f'{m}_ood_test']-w[f'{m}_id_test']
    w['condition_id']=w.domain_family.astype(str)+'|'+w.condition.astype(str)
    return w

def main():
    w=prepare(); w.to_csv(OUT/'cross_domain_shift_deltas.csv',index=False)
    x=w[w.model=='logistic_regression'].copy()
    rows=[]
    warnings.simplefilter('ignore',ConvergenceWarning)
    for m in METRICS:
        fit=smf.mixedlm(f'delta_{m} ~ 1',x,groups=x.domain_family,re_formula='1',vc_formula={'condition':'0 + C(condition_id)'}).fit(reml=True,method='lbfgs',maxiter=1000,disp=False)
        vd=float(fit.cov_re.iloc[0,0]); vc=float(fit.vcomp[0]); vr=float(fit.scale); total=vd+vc+vr
        rows.append({
            'metric':m,
            'n_delta_observations':len(x),
            'n_domains':x.domain_family.nunique(),
            'n_conditions':x.condition_id.nunique(),
            'mixed_model_intercept_mean_shift':float(fit.fe_params['Intercept']),
            'intercept_se':float(fit.bse_fe['Intercept']),
            'domain_variance':vd,'condition_within_domain_variance':vc,'refit_residual_variance':vr,
            'domain_variance_fraction':vd/total,'condition_variance_fraction':vc/total,'refit_variance_fraction':vr/total,
            'converged':bool(fit.converged)
        })
    vcdf=pd.DataFrame(rows); vcdf.to_csv(OUT/'cross_domain_variance_components.csv',index=False)

    # Descriptive model-family sensitivity of the shift delta, with logistic regression as reference.
    z=w.copy(); z['model']=pd.Categorical(z.model,categories=['logistic_regression','lightgbm'])
    effects=[]
    for m in METRICS:
        fit=smf.mixedlm(f'delta_{m} ~ C(model)',z,groups=z.domain_family,re_formula='1',vc_formula={'condition':'0 + C(condition_id)'}).fit(reml=True,method='lbfgs',maxiter=1000,disp=False)
        term='C(model)[T.lightgbm]'
        effects.append({
            'metric':m,
            'n_delta_observations':len(z),
            'logistic_reference_mean_shift':float(fit.fe_params['Intercept']),
            'lightgbm_minus_logistic_shift_delta':float(fit.fe_params[term]),
            'lightgbm_effect_se':float(fit.bse_fe[term]),
            'lightgbm_effect_p_descriptive':float(fit.pvalues[term]),
            'converged':bool(fit.converged)
        })
    eff=pd.DataFrame(effects); eff.to_csv(OUT/'model_family_delta_effects.csv',index=False)
    print(vcdf.to_string(index=False))
    print('\nModel-family shift effects:')
    print(eff.to_string(index=False))

if __name__=='__main__': main()
