import os
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from common import safe_auroc,safe_ap,expected_calibration_error,fit_calibrators,metrics,empirical_summary
ROOT=Path(os.environ.get('CRIT_AID_ROOT', Path(__file__).resolve().parents[1]));PREP=ROOT/'prepared';OUT=ROOT/'outputs';BASE=20260728;EPS=1e-8
CPI18=251.107;CPI24=313.689;TH24=50000*CPI24/CPI18
FEATURES=['AGEP','COW','SCHL','MAR','OCCP','POBP','REL_HARM','WKHP','SEX','RAC1P'];NUM=['AGEP','WKHP'];CAT=[c for c in FEATURES if c not in NUM]
TARGETS={'unadjusted_reported_income_50000':'target_unadjusted_50000','survey_year_adjusted_50000':'target_survey_year_adjusted_50000','cpi_constant_2018usd':'target_cpi_constant_2018usd'}
def pp():return ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),NUM),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',min_frequency=10,dtype=np.float32))]),CAT)],sparse_threshold=.3)
def split4(frame,seed):
 idx=np.arange(len(frame));s=frame.state.astype(str)+'_'+frame.target_unadjusted_50000.astype(str);tr,t=train_test_split(idx,test_size=.4,random_state=seed,stratify=s);ca,r=train_test_split(t,test_size=.625,random_state=seed+1,stratify=s.iloc[t]);cs,te=train_test_split(r,test_size=.6,random_state=seed+2,stratify=s.iloc[r]);return tr,ca,cs,te
# Fast paired target bootstrap on saved fixed-refit predictions.
t=pd.read_csv(OUT/'acs_primary_target_predictions.csv.gz');rng=np.random.default_rng(BASE+9000);strata=(t.state.astype(str)+'_'+t.y_unadjusted.astype(str)+'_'+t.y_cpi.astype(str)).to_numpy();groups=[np.flatnonzero(strata==s) for s in np.unique(strata)];rows=[]
def simple(y,p,w):
 from sklearn.metrics import brier_score_loss,log_loss
 return {'auroc':safe_auroc(y,p,w),'average_precision':safe_ap(y,p,w),'brier':brier_score_loss(y,p,sample_weight=w),'log_loss':log_loss(y,p,labels=[0,1],sample_weight=w),'ece':expected_calibration_error(y,p,w,15)[0]}
for b in range(500):
 idx=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups]);rng.shuffle(idx);w=t.weight.to_numpy(float)[idx]
 a=simple(t.y_unadjusted.to_numpy(int)[idx],t.p_unadjusted_reported_income_50000.to_numpy(float)[idx],w);c=simple(t.y_cpi.to_numpy(int)[idx],t.p_cpi_constant_2018usd.to_numpy(float)[idx],w)
 for m in a:rows.append({'bootstrap':b,'contrast':'unadjusted_minus_cpi','metric':m,'delta':a[m]-c[m]})
b=pd.DataFrame(rows);b.to_csv(OUT/'acs_target_definition_paired_bootstrap.csv.gz',index=False,compression='gzip');
# quantile summary
q=b.groupby(['contrast','metric']).delta.agg(mean='mean',sd='std',q025=lambda x:x.quantile(.025),median='median',q975=lambda x:x.quantile(.975),min='min',max='max').reset_index();q.to_csv(OUT/'acs_target_definition_paired_bootstrap_summary.csv',index=False)
# Five independent computational cohorts, one complete source split per cohort.
df=pd.read_csv(PREP/'acs_harmonized_with_ids.csv.gz');df['target_cpi_constant_2018usd']=np.where(df.year.eq(2018),df.PINCP_ADJ_YEAR_DOLLARS>50000,df.PINCP_ADJ_YEAR_DOLLARS>TH24).astype(np.int8);sf=df[df.year==2018].reset_index(drop=True);tf=df[df.year==2024].reset_index(drop=True);res=[];coh=[]
for cr in range(5):
 seed=BASE+7000+cr*503;ss=sf.state.astype(str)+'_'+sf.target_unadjusted_50000.astype(str);ts=tf.state.astype(str)+'_'+tf.target_unadjusted_50000.astype(str);si,_=train_test_split(np.arange(len(sf)),train_size=100000,random_state=seed,stratify=ss);ti,_=train_test_split(np.arange(len(tf)),train_size=75000,random_state=seed+1,stratify=ts);s=sf.loc[si].reset_index(drop=True);o=tf.loc[ti].reset_index(drop=True);coh.append(pd.concat([s[['record_key']].assign(cohort_repeat=cr,domain='source'),o[['record_key']].assign(cohort_repeat=cr,domain='target')]))
 tr,ca,cs,te=split4(s,seed+2);prep=pp();Xtr=prep.fit_transform(s.loc[tr,FEATURES]);Xca=prep.transform(s.loc[ca,FEATURES]);Xid=prep.transform(s.loc[te,FEATURES]);Xo=prep.transform(o[FEATURES]);wtr=s.loc[tr,'PWGTP'].to_numpy(float);wca=s.loc[ca,'PWGTP'].to_numpy(float);wid=s.loc[te,'PWGTP'].to_numpy(float);wo=o.PWGTP.to_numpy(float)
 for tn,col in TARGETS.items():
  ys=s[col].to_numpy(int);yo=o[col].to_numpy(int)
  model=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,tol=1e-6).fit(Xtr,ys[tr],sample_weight=wtr/wtr.mean());cal=fit_calibrators(ys[ca],model.predict_proba(Xca)[:,1],wca/wca.mean())['platt'];pi=cal.predict(model.predict_proba(Xid)[:,1]);po=cal.predict(model.predict_proba(Xo)[:,1])
  for dom,y,p,w in [('id_test',ys[te],pi,wid),('ood_test',yo,po,wo)]:res.append({'cohort_repeat':cr,'seed':seed,'target_definition':tn,'model':'logistic_regression','mapping':'platt','domain':dom,**metrics(y,p,w)})
 print('ACS independent cohort',cr,'done',flush=True)
pd.DataFrame(res).to_csv(OUT/'acs_independent_cohort_sensitivity.csv',index=False);pd.concat(coh,ignore_index=True).to_csv(ROOT/'manifests'/'acs_independent_cohort_manifest.csv.gz',index=False,compression='gzip')
empirical_summary(pd.DataFrame(res),['target_definition','domain'],['prevalence','auroc','average_precision','brier','log_loss','ece','calibration_intercept','calibration_slope','hcep_090','aurc']).to_csv(OUT/'acs_independent_cohort_sensitivity_summary.csv',index=False)
meta={'base_seed':BASE,'repeats':10,'independent_cohort_repeats':5,'source_eligible_n':len(sf),'target_eligible_n':len(tf),'source_computational_n':100000,'target_computational_n':75000,'cpi_2024_equivalent_threshold':TH24,'paired_bootstrap_repeats':500};(OUT/'acs_metadata.json').write_text(json.dumps(meta,indent=2))
print('ACS postprocess complete')
