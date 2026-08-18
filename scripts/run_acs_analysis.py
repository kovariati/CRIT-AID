from __future__ import annotations

import os

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    conformal_metrics, empirical_summary, expected_calibration_error, fit_calibrators,
    label_conditional_qhat, marginal_qhat, metrics, risk_selection_curve,
    save_environment, selection_threshold, selective_metrics, weighted_mean,
)

ROOT=Path(os.environ.get('CRIT_AID_ROOT', Path(__file__).resolve().parents[1])); PREP=ROOT/'prepared'; OUT=ROOT/'outputs'; MAN=ROOT/'manifests'
OUT.mkdir(exist_ok=True); MAN.mkdir(exist_ok=True)
DATA=PREP/'acs_harmonized_with_ids.csv.gz'
BASE=20260728; NREP=10; EPS=1e-8
CPI18=251.107; CPI24=313.689; TH24=50000*CPI24/CPI18
FEATURES=['AGEP','COW','SCHL','MAR','OCCP','POBP','REL_HARM','WKHP','SEX','RAC1P']
NUM=['AGEP','WKHP']; CAT=[c for c in FEATURES if c not in NUM]
TARGETS={
    'unadjusted_reported_income_50000':'target_unadjusted_50000',
    'survey_year_adjusted_50000':'target_survey_year_adjusted_50000',
    'cpi_constant_2018usd':'target_cpi_constant_2018usd',
}
SELECTION_RATES=[0.50,0.70,0.80,0.90,0.95]
ALPHAS=[0.05,0.10,0.20]


def preprocessor():
    return ColumnTransformer([
        ('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),NUM),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',min_frequency=10,dtype=np.float32))]),CAT),
    ],sparse_threshold=.3)


def split4(frame,seed):
    idx=np.arange(len(frame)); strata=frame.state.astype(str)+'_'+frame.target_unadjusted_50000.astype(str)
    tr,temp=train_test_split(idx,test_size=.40,random_state=seed,stratify=strata)
    ca,rem=train_test_split(temp,test_size=.625,random_state=seed+1,stratify=strata.iloc[temp])
    cs,te=train_test_split(rem,test_size=.60,random_state=seed+2,stratify=strata.iloc[rem])
    return tr,ca,cs,te


def models(seed):
    return {
        'logistic_regression':LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000,tol=1e-6),
        'lightgbm':LGBMClassifier(n_estimators=200,learning_rate=.05,num_leaves=31,min_child_samples=50,subsample=.9,colsample_bytree=.9,reg_lambda=1.0,random_state=seed,n_jobs=-1,verbosity=-1),
    }


def bootstrap_paired(target,preds,n_boot=500,seed=BASE+9000):
    # preds: dict target_name -> p for the same target rows; survey-weighted record bootstrap.
    rng=np.random.default_rng(seed)
    strata=(target.state.astype(str)+'_'+target.target_unadjusted_50000.astype(str)+'_'+target.target_cpi_constant_2018usd.astype(str)).to_numpy()
    groups={s:np.flatnonzero(strata==s) for s in np.unique(strata)}
    y_u=target.target_unadjusted_50000.to_numpy(int); y_c=target.target_cpi_constant_2018usd.to_numpy(int); w=target.PWGTP.to_numpy(float)
    rows=[]
    for b in range(n_boot):
        idx=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups.values()]); rng.shuffle(idx)
        mu=metrics(y_u[idx],preds['unadjusted_reported_income_50000'][idx],w[idx])
        mc=metrics(y_c[idx],preds['cpi_constant_2018usd'][idx],w[idx])
        for m in ['auroc','brier','log_loss','ece','calibration_intercept','calibration_slope','coverage_marginal']:
            if m=='coverage_marginal': continue
            rows.append({'bootstrap':b,'contrast':'unadjusted_minus_cpi','metric':m,'delta':mu[m]-mc[m]})
    return pd.DataFrame(rows)


def main():
    t0=time.time(); df=pd.read_csv(DATA)
    df['target_cpi_constant_2018usd']=np.where(df.year.eq(2018),df.PINCP_ADJ_YEAR_DOLLARS>50000,df.PINCP_ADJ_YEAR_DOLLARS>TH24).astype(np.int8)
    source_full=df[df.year.eq(2018)].reset_index(drop=True); target_full=df[df.year.eq(2024)].reset_index(drop=True)
    source_strata=source_full.state.astype(str)+'_'+source_full.target_unadjusted_50000.astype(str)
    target_strata=target_full.state.astype(str)+'_'+target_full.target_unadjusted_50000.astype(str)
    si,_=train_test_split(np.arange(len(source_full)),train_size=100000,random_state=BASE,stratify=source_strata)
    ti,_=train_test_split(np.arange(len(target_full)),train_size=75000,random_state=BASE+1,stratify=target_strata)
    source=source_full.loc[si].reset_index(drop=True); target=target_full.loc[ti].reset_index(drop=True)
    pd.concat([
        source[['record_key','year','state']].assign(cohort_domain='source_computational'),
        target[['record_key','year','state']].assign(cohort_domain='target_computational')
    ],ignore_index=True).to_csv(MAN/'acs_cohort_manifest.csv.gz',index=False,compression='gzip')

    metric_rows=[]; selection_rows=[]; conformal_rows=[]; curve_rows=[]; split_rows=[]; conv_rows=[]; primary_predictions={}
    for rep in range(NREP):
        seed=BASE+101*rep; tr,ca,cs,te=split4(source,seed)
        assignment=np.full(len(source),'',object); assignment[tr]='train';assignment[ca]='probability_calibration';assignment[cs]='selection_conformal_calibration';assignment[te]='id_test'
        split_rows.append(pd.DataFrame({'record_key':source.record_key,'repeat':rep,'seed':seed,'partition':assignment}))
        pp=preprocessor(); Xtr=pp.fit_transform(source.loc[tr,FEATURES]);Xca=pp.transform(source.loc[ca,FEATURES]);Xcs=pp.transform(source.loc[cs,FEATURES]);Xid=pp.transform(source.loc[te,FEATURES]);Xood=pp.transform(target[FEATURES])
        wtr=source.loc[tr,'PWGTP'].to_numpy(float);wca=source.loc[ca,'PWGTP'].to_numpy(float);wcs=source.loc[cs,'PWGTP'].to_numpy(float);wid=source.loc[te,'PWGTP'].to_numpy(float);wood=target.PWGTP.to_numpy(float)
        for target_name,col in TARGETS.items():
            ys=source[col].to_numpy(int); yo=target[col].to_numpy(int)
            for model_name,model in models(seed).items():
                model.fit(Xtr,ys[tr],sample_weight=wtr/wtr.mean())
                n_iter=(int(np.max(np.atleast_1d(model.n_iter_))) if hasattr(model,'n_iter_') else np.nan)
                pca_raw=model.predict_proba(Xca)[:,1];pcs_raw=model.predict_proba(Xcs)[:,1];pid_raw=model.predict_proba(Xid)[:,1];pood_raw=model.predict_proba(Xood)[:,1]
                calibrators=fit_calibrators(ys[ca],pca_raw,wca/wca.mean())
                for mapping,cal in calibrators.items():
                    if cal is None: continue
                    pcs=np.clip(cal.predict(pcs_raw),EPS,1-EPS);pid=np.clip(cal.predict(pid_raw),EPS,1-EPS);pood=np.clip(cal.predict(pood_raw),EPS,1-EPS)
                    conv_rows.append({'repeat':rep,'target_definition':target_name,'model':model_name,'mapping':mapping,'model_iterations':n_iter,'calibrator_iterations':getattr(cal,'n_iter_',0),'model_converged':True,'calibrator_converged':True})
                    if rep==0 and model_name=='logistic_regression' and mapping=='platt': primary_predictions[target_name]=pood.copy()
                    for domain,y,p,w in [('id_test',ys[te],pid,wid),('ood_test',yo,pood,wood)]:
                        base={'repeat':rep,'seed':seed,'target_definition':target_name,'model':model_name,'mapping':mapping,'domain':domain,'weighting':'survey_weighted'}
                        metric_rows.append({**base,**metrics(y,p,w)})
                        base_u={**base,'weighting':'unweighted'};metric_rows.append({**base_u,**metrics(y,p,None)})
                    conf_cal=np.maximum(pcs,1-pcs)
                    for desired in SELECTION_RATES:
                        thr=selection_threshold(conf_cal,desired,wcs)
                        for domain,y,p,w in [('id_test',ys[te],pid,wid),('ood_test',yo,pood,wood)]:
                            base={'repeat':rep,'target_definition':target_name,'model':model_name,'mapping':mapping,'domain':domain,'desired_selection_rate':desired,'source_threshold':thr,'weighting':'survey_weighted'}
                            selection_rows.append({**base,**selective_metrics(y,p,thr,w)})
                        # Save risk-selection curves only for primary mapping/model and 0.80 not needed; full curve independent of target desired.
                    if model_name=='logistic_regression' and mapping=='platt':
                        for domain,y,p,w in [('id_test',ys[te],pid,wid),('ood_test',yo,pood,wood)]:
                            curve,_,_=risk_selection_curve(y,p,w);curve['repeat']=rep;curve['target_definition']=target_name;curve['domain']=domain;curve_rows.append(curve)
                    for alpha in ALPHAS:
                        qm,k,n=marginal_qhat(ys[cs],pcs,alpha); qlc=label_conditional_qhat(ys[cs],pcs,alpha)
                        for method,qobj in [('marginal',qm),('label_conditional',qlc)]:
                            for domain,y,p,w in [('id_test',ys[te],pid,wid),('ood_test',yo,pood,wood)]:
                                for weighting,ww in [('survey_weighted',w),('unweighted',None)]:
                                    base={'repeat':rep,'target_definition':target_name,'model':model_name,'mapping':mapping,'domain':domain,'alpha':alpha,'nominal_coverage':1-alpha,'conformal_method':method,'weighting':weighting,'marginal_rank_k':k,'marginal_calibration_n':n,'class0_rank_k':qlc[0]['k'],'class0_calibration_n':qlc[0]['n'],'class1_rank_k':qlc[1]['k'],'class1_calibration_n':qlc[1]['n']}
                                    conformal_rows.append({**base,**conformal_metrics(y,p,qobj,ww,method=method)})
        print('ACS repeat',rep,'done',flush=True)
    metrics_df=pd.DataFrame(metric_rows);sel_df=pd.DataFrame(selection_rows);conf_df=pd.DataFrame(conformal_rows);curves=pd.concat(curve_rows,ignore_index=True);splits=pd.concat(split_rows,ignore_index=True);conv=pd.DataFrame(conv_rows)
    metrics_df.to_csv(OUT/'acs_metrics_complete.csv.gz',index=False,compression='gzip');sel_df.to_csv(OUT/'acs_selection_complete.csv.gz',index=False,compression='gzip');conf_df.to_csv(OUT/'acs_conformal_complete.csv.gz',index=False,compression='gzip');curves.to_csv(OUT/'acs_risk_selection_curves.csv.gz',index=False,compression='gzip');splits.to_csv(MAN/'acs_split_manifest.csv.gz',index=False,compression='gzip');conv.to_csv(OUT/'acs_convergence_log.csv',index=False)
    metric_cols=['prevalence','auroc','average_precision','pr_skill','brier','log_loss','ece','calibration_intercept','calibration_slope','hcep_080','hcep_090','hcep_095','aurc','excess_aurc']
    empirical_summary(metrics_df,['target_definition','model','mapping','domain','weighting'],metric_cols).to_csv(OUT/'acs_metrics_summary.csv',index=False)
    empirical_summary(sel_df,['target_definition','model','mapping','domain','desired_selection_rate','weighting'],['selection_rate','selective_risk','selective_accuracy']).to_csv(OUT/'acs_selection_summary.csv',index=False)
    empirical_summary(conf_df,['target_definition','model','mapping','domain','alpha','conformal_method','weighting'],['coverage','avg_set_size','singleton_rate','ambiguous_rate','empty_rate','singleton_accuracy','class0_coverage','class1_coverage','class0_avg_set_size','class1_avg_set_size','class0_singleton_accuracy','class1_singleton_accuracy']).to_csv(OUT/'acs_conformal_summary.csv',index=False)

    # Paired ID-to-OOD differences for primary model/mapping.
    prim=metrics_df[(metrics_df.model=='logistic_regression')&(metrics_df.mapping=='platt')&(metrics_df.weighting=='survey_weighted')]
    wide=prim.pivot(index=['repeat','target_definition'],columns='domain',values=metric_cols).reset_index()
    deltas=[]
    for _,r in wide.iterrows():
        for m in metric_cols: deltas.append({'repeat':int(r['repeat']),'target_definition':r['target_definition'],'metric':m,'delta_ood_minus_id':r[(m,'ood_test')]-r[(m,'id_test')]})
    pd.DataFrame(deltas).to_csv(OUT/'acs_paired_id_ood_differences.csv',index=False)

    # Primary fixed-refit subgroup diagnostics and reliability bins for CPI target.
    p=primary_predictions['cpi_constant_2018usd'];y=target.target_cpi_constant_2018usd.to_numpy(int);w=target.PWGTP.to_numpy(float)
    groups={'state':target.state.astype(str),'sex':target.SEX.map({1:'Male',2:'Female'}).fillna('Other').astype(str),'race':target.RAC1P.map(lambda x:'White alone' if x==1 else ('Black or African American alone' if x==2 else ('Asian alone' if x==6 else 'Other pooled (RAC1P 3-5,7-9)')))}
    sub=[]
    for gt,gv in groups.items():
        for lev in sorted(gv.unique()):
            m=(gv==lev).to_numpy()
            if m.sum()>=200 and len(np.unique(y[m]))==2:
                qlc=label_conditional_qhat(y,p,.10) # descriptive target-label oracle only not used as deployment method
                sub.append({'group_type':gt,'group':lev,**metrics(y[m],p[m],w[m])})
    pd.DataFrame(sub).to_csv(OUT/'acs_primary_subgroup_metrics.csv',index=False)
    _,bins=expected_calibration_error(y,p,w,15);bins.to_csv(OUT/'acs_primary_reliability_bins.csv',index=False)
    pd.DataFrame({'record_key':target.record_key,'state':target.state,'y_unadjusted':target.target_unadjusted_50000,'y_adjusted':target.target_survey_year_adjusted_50000,'y_cpi':target.target_cpi_constant_2018usd,'weight':target.PWGTP,**{f'p_{k}':v for k,v in primary_predictions.items()}}).to_csv(OUT/'acs_primary_target_predictions.csv.gz',index=False,compression='gzip')
    boot=bootstrap_paired(target,primary_predictions,500);boot.to_csv(OUT/'acs_target_definition_paired_bootstrap.csv.gz',index=False,compression='gzip');empirical_summary(boot,['contrast','metric'],['delta']).to_csv(OUT/'acs_target_definition_paired_bootstrap_summary.csv',index=False)

    # Full-eligible-sample fixed-split sensitivity, primary Platt model for both model families.
    full_rows=[]
    tr,ca,cs,te=split4(source_full,BASE+777)
    pp=preprocessor();Xtr=pp.fit_transform(source_full.loc[tr,FEATURES]);Xca=pp.transform(source_full.loc[ca,FEATURES]);Xcs=pp.transform(source_full.loc[cs,FEATURES]);Xid=pp.transform(source_full.loc[te,FEATURES]);Xood=pp.transform(target_full[FEATURES])
    wtr=source_full.loc[tr,'PWGTP'].to_numpy(float);wca=source_full.loc[ca,'PWGTP'].to_numpy(float);wid=source_full.loc[te,'PWGTP'].to_numpy(float);wood=target_full.PWGTP.to_numpy(float)
    for tn,col in TARGETS.items():
        ys=source_full[col].to_numpy(int);yo=target_full[col].to_numpy(int)
        for mn,model in models(BASE+777).items():
            model.fit(Xtr,ys[tr],sample_weight=wtr/wtr.mean());cal=fit_calibrators(ys[ca],model.predict_proba(Xca)[:,1],wca/wca.mean())['platt'];pid=cal.predict(model.predict_proba(Xid)[:,1]);po=cal.predict(model.predict_proba(Xood)[:,1])
            for dom,yv,pv,wv in [('id_test',ys[te],pid,wid),('ood_test',yo,po,wood)]:full_rows.append({'target_definition':tn,'model':mn,'mapping':'platt','domain':dom,**metrics(yv,pv,wv)})
        print('ACS full sample',tn,'done',flush=True)
    pd.DataFrame(full_rows).to_csv(OUT/'acs_full_sample_sensitivity.csv',index=False)
    save_environment(OUT/'acs_environment.json')
    (OUT/'acs_metadata.json').write_text(json.dumps({'base_seed':BASE,'repeats':NREP,'source_eligible_n':len(source_full),'target_eligible_n':len(target_full),'source_computational_n':len(source),'target_computational_n':len(target),'cpi_2024_equivalent_threshold':TH24,'runtime_seconds':time.time()-t0},indent=2),encoding='utf-8')
    print('ACS ANALYSIS COMPLETE',time.time()-t0,flush=True)

if __name__=='__main__': main()
