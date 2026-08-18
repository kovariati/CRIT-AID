from pathlib import Path
import os
import numpy as np, pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss
from common import expected_calibration_error
ROOT=Path(os.environ.get('CRIT_AID_ROOT', Path(__file__).resolve().parents[1])); OUT=ROOT/'outputs'
EPS=1e-8

def wm(x,w): return float(np.average(np.asarray(x,float),weights=np.asarray(w,float)))
def shift_to_prev(p,target,w):
    z=logit(np.clip(np.asarray(p,float),EPS,1-EPS)); w=np.asarray(w,float)
    f=lambda a: np.average(expit(z+a),weights=w)-target
    return float(brentq(f,-20,20))
def basic(y,p,w):
    return {'ece':expected_calibration_error(y,p,w,15)[0], 'brier':float(brier_score_loss(y,p,sample_weight=w)), 'log_loss':float(log_loss(y,np.clip(p,EPS,1-EPS),labels=[0,1],sample_weight=w))}
def main(n_boot=300):
    d=pd.read_csv(OUT/'acs_primary_target_predictions.csv.gz')
    yu=d.y_unadjusted.to_numpy(int); yc=d.y_cpi.to_numpy(int); w=d.weight.to_numpy(float)
    pu=d.p_unadjusted_reported_income_50000.to_numpy(float); pc=d.p_cpi_constant_2018usd.to_numpy(float)
    cells={
      'UU':basic(yu,pu,w), 'UC':basic(yu,pc,w), 'CU':basic(yc,pu,w), 'CC':basic(yc,pc,w)
    }
    rows=[]
    for m in ['ece','brier','log_loss']:
        total=cells['UU'][m]-cells['CC'][m]
        label=.5*((cells['UU'][m]-cells['CU'][m])+(cells['UC'][m]-cells['CC'][m]))
        pred=.5*((cells['UU'][m]-cells['UC'][m])+(cells['CU'][m]-cells['CC'][m]))
        rows.append({'metric':m,'UU':cells['UU'][m],'UC':cells['UC'][m],'CU':cells['CU'][m],'CC':cells['CC'][m],'total_unadjusted_minus_cpi':total,'prediction_pipeline_component':pred,'target_label_definition_component':label,'component_sum':pred+label})
    pd.DataFrame(rows).to_csv(OUT/'acs_two_factor_metric_decomposition.csv',index=False)
    # label flips / prevalence
    flip=(yu!=yc)
    prevalence=pd.DataFrame([{
      'weighted_prevalence_unadjusted':wm(yu,w),'weighted_prevalence_cpi':wm(yc,w),
      'weighted_prevalence_difference':wm(yu,w)-wm(yc,w),'weighted_label_flip_rate':wm(flip,w),
      'weighted_1_to_0_rate':wm((yu==1)&(yc==0),w),'weighted_0_to_1_rate':wm((yu==0)&(yc==1),w),
      'unweighted_label_flip_rate':float(flip.mean())
    }]); prevalence.to_csv(OUT/'acs_target_label_change_summary.csv',index=False)
    # full-sample alignment
    au=shift_to_prev(pu,wm(yu,w),w); ac=shift_to_prev(pc,wm(yc,w),w)
    pua=expit(logit(np.clip(pu,EPS,1-EPS))+au); pca=expit(logit(np.clip(pc,EPS,1-EPS))+ac)
    rawu=basic(yu,pu,w); rawc=basic(yc,pc,w); alu=basic(yu,pua,w); alc=basic(yc,pca,w)
    summary=[]
    for m in ['ece','brier','log_loss']:
      summary.append({'metric':m,'raw_unadjusted':rawu[m],'raw_cpi':rawc[m],'raw_difference':rawu[m]-rawc[m],'aligned_unadjusted':alu[m],'aligned_cpi':alc[m],'aligned_difference':alu[m]-alc[m]})
    # bootstrap
    strata=(d.state.astype(str)+'|'+d.y_unadjusted.astype(str)+'|'+d.y_cpi.astype(str)).to_numpy(); groups=[np.flatnonzero(strata==s) for s in np.unique(strata)]
    rng=np.random.default_rng(20260818); boot=[]
    for b in range(n_boot):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        yy_u,yc_b,ww,ppu,ppc=yu[idx],yc[idx],w[idx],pu[idx],pc[idx]
        su=shift_to_prev(ppu,wm(yy_u,ww),ww); sc=shift_to_prev(ppc,wm(yc_b,ww),ww)
        bu=basic(yy_u,expit(logit(np.clip(ppu,EPS,1-EPS))+su),ww); bc=basic(yc_b,expit(logit(np.clip(ppc,EPS,1-EPS))+sc),ww)
        for m in ['ece','brier','log_loss']: boot.append({'bootstrap':b,'metric':m,'aligned_difference':bu[m]-bc[m]})
    boot=pd.DataFrame(boot); boot.to_csv(OUT/'acs_prevalence_alignment_bootstrap.csv.gz',index=False,compression='gzip')
    for r in summary:
        vals=boot[boot.metric==r['metric']].aligned_difference.to_numpy(); r['aligned_ci025']=float(np.quantile(vals,.025)); r['aligned_ci975']=float(np.quantile(vals,.975))
    pd.DataFrame(summary).to_csv(OUT/'acs_prevalence_alignment_summary.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(prevalence.to_string(index=False)); print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__': main()
