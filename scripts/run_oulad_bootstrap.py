import os
from pathlib import Path
import numpy as np, pandas as pd
from common import safe_auroc, expected_calibration_error, empirical_summary
ROOT=Path(os.environ.get('CRIT_AID_ROOT', Path(__file__).resolve().parents[1]));OUT=ROOT/'outputs';BASE=20260728
p=pd.read_csv(OUT/'oulad_primary_pair_predictions.csv.gz')
rng=np.random.default_rng(BASE+8000);rows=[]
for horizon,g_h in p.groupby('horizon_day'):
    pdata={pair:g[['y','p','marginal_include0','marginal_include1']].to_numpy() for pair,g in g_h.groupby('analysis_id')}
    pairs=np.array(sorted(pdata))
    for b in range(500):
        vals=[]
        for pair in rng.choice(pairs,size=len(pairs),replace=True):
            a=pdata[pair]; z=a[rng.integers(0,len(a),size=len(a))]
            y=z[:,0].astype(int); pr=z[:,1].astype(float); i0=z[:,2].astype(bool);i1=z[:,3].astype(bool)
            au=safe_auroc(y,pr); ece=expected_calibration_error(y,pr,n_bins=15)[0]; cov=np.where(y==1,i1,i0).mean()
            vals.append((len(z),au,ece,cov))
        vals=np.array(vals,float)
        for weighting,w in [('equal_pair',np.ones(len(vals))),('size_weighted',vals[:,0])]:
            for j,met in enumerate(['auroc','ece','coverage'],start=1):rows.append({'bootstrap':b,'horizon_day':horizon,'weighting':weighting,'metric':met,'value':np.average(vals[:,j],weights=w)})
df=pd.DataFrame(rows);df.to_csv(OUT/'oulad_two_stage_bootstrap.csv.gz',index=False,compression='gzip')
empirical_summary(df,['horizon_day','weighting','metric'],['value']).to_csv(OUT/'oulad_two_stage_bootstrap_summary.csv',index=False)
print('fast OULAD bootstrap complete',df.shape)
