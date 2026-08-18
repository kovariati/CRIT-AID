from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (EPS, fit_calibrators, metrics, selection_threshold, selective_metrics,
                    marginal_qhat, label_conditional_qhat, conformal_metrics)

ROOT=Path(os.environ.get('CRIT_AID_ROOT', Path(__file__).resolve().parents[1]))
PREP=ROOT/'prepared'; OUT=ROOT/'outputs'
BASE=20260728

LGB_PARAMS=dict(n_estimators=200,learning_rate=.05,num_leaves=31,min_child_samples=50,
                subsample=.9,colsample_bytree=.9,reg_lambda=1.0,n_jobs=-1,verbosity=-1)

def lgb(seed): return LGBMClassifier(random_state=int(seed),**LGB_PARAMS)

def row_metrics(base, y, p, ycs, pcs, weights=None, weights_cs=None):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); pcs=np.clip(np.asarray(pcs,float),EPS,1-EPS)
    out={**base,**metrics(y,p,weights)}
    th=selection_threshold(np.maximum(pcs,1-pcs),.80,weights_cs)
    sm=selective_metrics(y,p,th,weights)
    out['selection_rate_080']=sm['selection_rate']; out['selective_risk_080']=sm['selective_risk']
    q,_,_=marginal_qhat(ycs,pcs,.10)
    cm=conformal_metrics(y,p,q,weights,method='marginal')
    out['marginal_coverage_090']=cm['coverage']; out['marginal_avg_set_size_090']=cm['avg_set_size']
    out['marginal_worst_class_coverage_090']=min(cm['class0_coverage'],cm['class1_coverage'])
    qlc=label_conditional_qhat(ycs,pcs,.10)
    cl=conformal_metrics(y,p,qlc,weights,method='label_conditional')
    out['lc_coverage_090']=cl['coverage']; out['lc_avg_set_size_090']=cl['avg_set_size']
    out['lc_worst_class_coverage_090']=min(cl['class0_coverage'],cl['class1_coverage'])
    return out

# ---------- ACS: extract already-computed LR and LightGBM grid ----------
def add_acs(rows):
    met=pd.read_csv(OUT/'acs_metrics_complete.csv.gz')
    sel=pd.read_csv(OUT/'acs_selection_complete.csv.gz')
    conf=pd.read_csv(OUT/'acs_conformal_complete.csv.gz')
    keep=met[(met.model.isin(['logistic_regression','lightgbm']))&(met.mapping=='platt')&(met.weighting=='survey_weighted')]
    for _,r in keep.iterrows():
        b={'domain_family':'ACS','condition':r.target_definition,'repeat':int(r['repeat']),'seed':BASE+int(r['repeat']),
           'model':r.model,'domain':r.domain}
        o={**b, **{k:r[k] for k in ['prevalence','auroc','brier','log_loss','ece']}}
        ss=sel[(sel.repeat==r['repeat'])&(sel.target_definition==r.target_definition)&(sel.model==r.model)&(sel.mapping=='platt')&(sel.domain==r.domain)&(sel.weighting=='survey_weighted')&(np.isclose(sel.desired_selection_rate,.8))].iloc[0]
        o['selection_rate_080']=ss.selection_rate; o['selective_risk_080']=ss.selective_risk
        cc=conf[(conf.repeat==r['repeat'])&(conf.target_definition==r.target_definition)&(conf.model==r.model)&(conf.mapping=='platt')&(conf.domain==r.domain)&(conf.weighting=='survey_weighted')&(np.isclose(conf.alpha,.1))]
        for meth,prefix in [('marginal','marginal'),('label_conditional','lc')]:
            z=cc[cc.conformal_method==meth].iloc[0]
            o[f'{prefix}_coverage_090']=z.coverage; o[f'{prefix}_avg_set_size_090']=z.avg_set_size
            o[f'{prefix}_worst_class_coverage_090']=min(z.class0_coverage,z.class1_coverage)
        rows.append(o)

# ---------- OULAD pooled primary ----------
BASE_CAT=['gender','region','highest_education','imd_band','age_band','disability']
def oulad_add_features(df):
    x=df.copy(); ad=x.active_days.fillna(0).clip(lower=0); total=x.clicks_total.fillna(0).clip(lower=0); rec=x.vle_record_count.fillna(0).clip(lower=0); h=x.horizon_day.astype(float).clip(lower=1)
    x['log_clicks_total']=np.log1p(total); x['log_vle_record_count']=np.log1p(rec); x['clicks_per_active_day']=total/(ad+1); x['records_per_active_day']=rec/(ad+1); x['active_day_fraction']=(ad/(h+1)).clip(0,1); x['recency_fraction']=(x.days_since_last_activity.fillna(h+1)/(h+1)).clip(0,2); x['registration_lead_days']=(-x.date_registration.fillna(0)).clip(-200,400); x['presentation_period']=x.code_presentation.str[-1]
    resource=[c for c in x.columns if c.startswith('clicks_') and c!='clicks_total' and not c.endswith('_share')]
    for c in resource:x[c+'_share']=x[c].fillna(0)/(total+1)
    return x,resource

def grouped_split(source,seed):
    g=source.groupby('id_student',as_index=False).target_unsuccessful.max()
    tr_ids,temp_ids=train_test_split(g.id_student,test_size=.40,random_state=seed,stratify=g.target_unsuccessful)
    temp=g[g.id_student.isin(temp_ids)]
    ca_ids,rem_ids=train_test_split(temp.id_student,test_size=.625,random_state=seed+1,stratify=temp.target_unsuccessful)
    rem=temp[temp.id_student.isin(rem_ids)]
    cs_ids,te_ids=train_test_split(rem.id_student,test_size=.60,random_state=seed+2,stratify=rem.target_unsuccessful)
    return source.id_student.isin(tr_ids).to_numpy(),source.id_student.isin(ca_ids).to_numpy(),source.id_student.isin(cs_ids).to_numpy(),source.id_student.isin(te_ids).to_numpy()

def pp(num,cat,minfreq=3):
    return ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',min_frequency=minfreq,dtype=np.float32))]),cat)],sparse_threshold=.3)

def matched_pairs(df):
    out=[]
    for m in sorted(df.code_module.unique()):
        for p in ['B','J']:
            if ((df.code_module==m)&(df.code_presentation==f'2013{p}')).any() and ((df.code_module==m)&(df.code_presentation==f'2014{p}')).any():out.append((m,p))
    return out

def add_oulad(rows):
    df,res=oulad_add_features(pd.read_csv(PREP/'oulad_all_registrations_horizons.csv.gz')); pairs=matched_pairs(df)
    behavior=['num_of_prev_attempts','studied_credits','registration_lead_days','log_clicks_total','log_vle_record_count','clicks_per_active_day','records_per_active_day','active_day_fraction','recency_fraction','no_vle_activity']+[c+'_share' for c in res]
    # logistic rows from existing results
    old=pd.read_csv(OUT/'oulad_metrics_complete.csv.gz')
    oldc=pd.read_csv(OUT/'oulad_conformal_complete.csv.gz'); olds=pd.read_csv(OUT/'oulad_selection_complete.csv.gz')
    filt=old[(old.analysis_id=='pooled_fixed_effects')&(old.estimand=='landmark_still_registered')&(old.representation=='score_free')]
    for _,r in filt.iterrows():
        b={'domain_family':'OULAD','condition':f'pooled_landmark_day{int(r.horizon_day)}','repeat':int(r['repeat']),'seed':int(r['seed']),'model':'logistic_regression','domain':r.domain}
        o={**b,**{k:r[k] for k in ['prevalence','auroc','brier','log_loss','ece']}}
        s=olds[(olds.analysis_id=='pooled_fixed_effects')&(olds.estimand=='landmark_still_registered')&(olds.representation=='score_free')&(olds.horizon_day==r.horizon_day)&(olds.repeat==r['repeat'])&(olds.domain==r.domain)&np.isclose(olds.desired_selection_rate,.8)].iloc[0]
        o['selection_rate_080']=s.selection_rate;o['selective_risk_080']=s.selective_risk
        c=oldc[(oldc.analysis_id=='pooled_fixed_effects')&(oldc.estimand=='landmark_still_registered')&(oldc.representation=='score_free')&(oldc.horizon_day==r.horizon_day)&(oldc.repeat==r['repeat'])&(oldc.domain==r.domain)&np.isclose(oldc.alpha,.1)]
        for meth,prefix in [('marginal','marginal'),('label_conditional','lc')]:
            z=c[c.conformal_method==meth].iloc[0];o[f'{prefix}_coverage_090']=z.coverage;o[f'{prefix}_avg_set_size_090']=z.avg_set_size;o[f'{prefix}_worst_class_coverage_090']=min(z.class0_coverage,z.class1_coverage)
        rows.append(o)
    # LightGBM reruns
    for h in [14,56]:
        dh=df[(df.horizon_day==h)&(df.registered_by_horizon==1)&(df.withdrawn_by_horizon==0)].copy()
        src=pd.concat([dh[(dh.code_module==m)&(dh.code_presentation==f'2013{p}')] for m,p in pairs],ignore_index=True)
        tgt=pd.concat([dh[(dh.code_module==m)&(dh.code_presentation==f'2014{p}')] for m,p in pairs],ignore_index=True)
        overlap=set(src.id_student.unique())&set(tgt.id_student.unique()); tgt=tgt[~tgt.id_student.isin(overlap)].copy()
        cols=behavior+BASE_CAT+['code_module','presentation_period']
        for rep in range(10):
            seed=BASE+1009*rep+h+73
            tr,ca,cs,te=grouped_split(src,seed); prep=pp(behavior,BASE_CAT+['code_module','presentation_period'],3)
            Xtr=prep.fit_transform(src.loc[tr,cols]);Xca=prep.transform(src.loc[ca,cols]);Xcs=prep.transform(src.loc[cs,cols]);Xid=prep.transform(src.loc[te,cols]);Xood=prep.transform(tgt[cols])
            y=src.target_unsuccessful.to_numpy(int);yo=tgt.target_unsuccessful.to_numpy(int)
            m=lgb(seed);m.fit(Xtr,y[tr]);cal=fit_calibrators(y[ca],m.predict_proba(Xca)[:,1])['platt'];pcs=cal.predict(m.predict_proba(Xcs)[:,1]);pid=cal.predict(m.predict_proba(Xid)[:,1]);pood=cal.predict(m.predict_proba(Xood)[:,1])
            for dom,yy,p in [('id_test',y[te],pid),('ood_test',yo,pood)]:
                rows.append(row_metrics({'domain_family':'OULAD','condition':f'pooled_landmark_day{h}','repeat':rep,'seed':seed,'model':'lightgbm','domain':dom},yy,p,y[cs],pcs))
        print('LightGBM OULAD',h,'done',flush=True)

# ---------- South German Credit ----------
SNUM=['laufzeit','hoehe','alter'];SCAT=['laufkont','moral','verw','sparkont','beszeit','rate','famges','buerge','wohnzeit','verm','weitkred','wohn','bishkred','beruf','pers','telef','gastarb'];SFEAT=SNUM+SCAT

def split4(y,seed,strata=None):
    y=np.asarray(y,int);idx=np.arange(len(y));strat=y if strata is None else np.asarray(strata)
    tr,temp=train_test_split(idx,test_size=.40,random_state=seed,stratify=strat); st=y[temp] if strata is None else np.asarray(strata)[temp]
    ca,rem=train_test_split(temp,test_size=.625,random_state=seed+1,stratify=st); sr=y[rem] if strata is None else np.asarray(strata)[rem]
    cs,te=train_test_split(rem,test_size=.60,random_state=seed+2,stratify=sr);return tr,ca,cs,te

def ext_pp(num,cat):
    return ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',min_frequency=2,dtype=np.float32))]),cat)],sparse_threshold=.3)

def rank_south(train,y,seed):
    z=pd.DataFrame(index=train.index);disc=[]
    for c in SFEAT:
        if c in SCAT:z[c]=pd.factorize(train[c].astype(str),sort=True)[0];disc.append(True)
        else:
            v=pd.to_numeric(train[c],errors='coerce');z[c]=v.fillna(v.median());disc.append(False)
    mi=mutual_info_classif(z[SFEAT],y,discrete_features=disc,random_state=seed);return [SFEAT[i] for i in np.argsort(-mi,kind='mergesort')]

def add_south(rows):
    d=pd.read_csv(PREP/'south_german_with_ids.csv.gz');d['target_bad_credit']=(d['kredit']==0).astype(int)
    for c in SCAT:d[c]=d[c].astype('object')
    y=d.target_bad_credit.to_numpy(int)
    # existing LR clean -> id, targeted -> ood
    met=pd.read_csv(OUT/'south_metrics_complete.csv.gz'); con=pd.read_csv(OUT/'south_conformal_complete.csv.gz'); sel=pd.read_csv(OUT/'south_selection_complete.csv.gz')
    for scenario,dom in [('clean','id_test'),('targeted_top3','ood_test')]:
        for _,r in met[(met.mapping=='platt')&(met.scenario==scenario)].iterrows():
            b={'domain_family':'SouthGermanCredit','condition':'targeted_top3','repeat':int(r['repeat']),'seed':int(r['seed']),'model':'logistic_regression','domain':dom}
            o={**b,**{k:r[k] for k in ['prevalence','auroc','brier','log_loss','ece']}}
            s=sel[(sel.repeat==r['repeat'])&(sel.mapping=='platt')&(sel.scenario==scenario)&np.isclose(sel.desired_selection_rate,.8)].iloc[0];o['selection_rate_080']=s.selection_rate;o['selective_risk_080']=s.selective_risk
            c=con[(con.repeat==r['repeat'])&(con.mapping=='platt')&(con.scenario==scenario)&np.isclose(con.alpha,.1)]
            for meth,prefix in [('marginal','marginal'),('label_conditional','lc')]:
                z=c[c.conformal_method==meth].iloc[0];o[f'{prefix}_coverage_090']=z.coverage;o[f'{prefix}_avg_set_size_090']=z.avg_set_size;o[f'{prefix}_worst_class_coverage_090']=min(z.class0_coverage,z.class1_coverage)
            rows.append(o)
    # LightGBM rerun
    for rep in range(20):
        seed=BASE+131*rep;tr,ca,cs,te=split4(y,seed);train=d.loc[tr].reset_index(drop=True);cald=d.loc[ca].reset_index(drop=True);confd=d.loc[cs].reset_index(drop=True);test=d.loc[te].reset_index(drop=True)
        rank=rank_south(train,train.target_bad_credit.to_numpy(int),seed); prep=ext_pp(SNUM,SCAT)
        Xtr=prep.fit_transform(train[SFEAT]);Xca=prep.transform(cald[SFEAT]);Xcs=prep.transform(confd[SFEAT])
        m=lgb(seed);m.fit(Xtr,train.target_bad_credit);cal=fit_calibrators(cald.target_bad_credit,m.predict_proba(Xca)[:,1])['platt'];pcs=cal.predict(m.predict_proba(Xcs)[:,1]);yt=test.target_bad_credit.to_numpy(int)
        clean=test[SFEAT].copy();targeted=test[SFEAT].copy();targeted.loc[:,rank[:3]]=np.nan
        for dom,x in [('id_test',clean),('ood_test',targeted)]:
            p=cal.predict(m.predict_proba(prep.transform(x))[:,1]); rows.append(row_metrics({'domain_family':'SouthGermanCredit','condition':'targeted_top3','repeat':rep,'seed':seed,'model':'lightgbm','domain':dom},yt,p,confd.target_bad_credit.to_numpy(int),pcs))
    print('LightGBM South German Credit done',flush=True)

# ---------- Heart ----------
HNUM=['age','resting_bp','cholesterol','max_heart_rate','oldpeak','major_vessels'];HCAT=['sex','chest_pain_type','fasting_blood_sugar','resting_ecg','exercise_angina','slope','thal'];HFEAT=HNUM+HCAT;SITES=['Cleveland','Hungary','Switzerland','VA Long Beach']
def add_heart(rows):
    d=pd.read_csv(PREP/'heart_disease_with_ids.csv.gz')
    for c in HCAT:d[c]=d[c].astype('object')
    met=pd.read_csv(OUT/'heart_metrics_complete.csv.gz');con=pd.read_csv(OUT/'heart_conformal_complete.csv.gz');sel=pd.read_csv(OUT/'heart_selection_complete.csv.gz')
    old=met[(met.mapping=='platt')&(met.missing_indicators==False)]
    for _,r in old.iterrows():
        b={'domain_family':'HeartDisease','condition':r.heldout_site,'repeat':int(r['repeat']),'seed':int(r['seed']),'model':'logistic_regression','domain':r.domain};o={**b,**{k:r[k] for k in ['prevalence','auroc','brier','log_loss','ece']}}
        s=sel[(sel.heldout_site==r.heldout_site)&(sel.repeat==r['repeat'])&(sel.mapping=='platt')&(sel.missing_indicators==False)&(sel.domain==r.domain)&np.isclose(sel.desired_selection_rate,.8)].iloc[0];o['selection_rate_080']=s.selection_rate;o['selective_risk_080']=s.selective_risk
        c=con[(con.heldout_site==r.heldout_site)&(con.repeat==r['repeat'])&(con.mapping=='platt')&(con.missing_indicators==False)&(con.domain==r.domain)&np.isclose(con.alpha,.1)]
        for meth,prefix in [('marginal','marginal'),('label_conditional','lc')]:
            z=c[c.conformal_method==meth].iloc[0];o[f'{prefix}_coverage_090']=z.coverage;o[f'{prefix}_avg_set_size_090']=z.avg_set_size;o[f'{prefix}_worst_class_coverage_090']=min(z.class0_coverage,z.class1_coverage)
        rows.append(o)
    for held in SITES:
        src=d[d.site!=held].reset_index(drop=True);tgt=d[d.site==held].reset_index(drop=True);ys=src.target_disease_present.to_numpy(int);yt=tgt.target_disease_present.to_numpy(int);strata=src.site.astype(str)+'|'+src.target_disease_present.astype(str)
        for rep in range(10):
            seed=BASE+149*rep+SITES.index(held);tr,ca,cs,te=split4(ys,seed,strata);prep=ext_pp(HNUM,HCAT)
            Xtr=prep.fit_transform(src.loc[tr,HFEAT]);Xca=prep.transform(src.loc[ca,HFEAT]);Xcs=prep.transform(src.loc[cs,HFEAT]);Xid=prep.transform(src.loc[te,HFEAT]);Xood=prep.transform(tgt[HFEAT])
            m=lgb(seed);m.fit(Xtr,ys[tr]);cal=fit_calibrators(ys[ca],m.predict_proba(Xca)[:,1])['platt'];pcs=cal.predict(m.predict_proba(Xcs)[:,1]);pid=cal.predict(m.predict_proba(Xid)[:,1]);pood=cal.predict(m.predict_proba(Xood)[:,1])
            for dom,yy,p in [('id_test',ys[te],pid),('ood_test',yt,pood)]:rows.append(row_metrics({'domain_family':'HeartDisease','condition':held,'repeat':rep,'seed':seed,'model':'lightgbm','domain':dom},yy,p,ys[cs],pcs))
        print('LightGBM Heart',held,'done',flush=True)

def main():
    rows=[];add_acs(rows);add_oulad(rows);add_south(rows);add_heart(rows)
    df=pd.DataFrame(rows).sort_values(['domain_family','condition','repeat','model','domain']).reset_index(drop=True)
    df.to_csv(OUT/'model_family_sensitivity_complete.csv.gz',index=False,compression='gzip')
    metrics_cols=['prevalence','auroc','brier','log_loss','ece','selection_rate_080','selective_risk_080','marginal_coverage_090','marginal_avg_set_size_090','marginal_worst_class_coverage_090','lc_coverage_090','lc_avg_set_size_090','lc_worst_class_coverage_090']
    agg=df.groupby(['domain_family','condition','model','domain'])[metrics_cols].agg(['mean','std']).reset_index()
    agg.columns=['_'.join([str(x) for x in c if x!='']).rstrip('_') if isinstance(c,tuple) else c for c in agg.columns]
    agg.to_csv(OUT/'model_family_sensitivity_summary.csv',index=False)
    print('rows',len(df),'written',flush=True)
if __name__=='__main__':main()
