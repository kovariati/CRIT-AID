from __future__ import annotations

import os

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    conformal_metrics, empirical_summary, fit_calibrators, label_conditional_qhat,
    marginal_qhat, metrics, risk_selection_curve, save_environment,
    selection_threshold, selective_metrics,
)

ROOT=Path(os.environ.get('CRIT_AID_ROOT', Path(__file__).resolve().parents[1]));PREP=ROOT/'prepared';OUT=ROOT/'outputs';MAN=ROOT/'manifests'
DATA=PREP/'oulad_all_registrations_horizons.csv.gz'
BASE=20260728;NREP=10;HORIZONS=[14,56];SELECTION_RATES=[.50,.70,.80,.90,.95];ALPHAS=[.05,.10,.20];EPS=1e-8
BASE_CAT=['gender','region','highest_education','imd_band','age_band','disability']


def add_features(df):
    x=df.copy();ad=x.active_days.fillna(0).clip(lower=0);total=x.clicks_total.fillna(0).clip(lower=0);rec=x.vle_record_count.fillna(0).clip(lower=0);h=x.horizon_day.astype(float).clip(lower=1)
    x['log_clicks_total']=np.log1p(total);x['log_vle_record_count']=np.log1p(rec);x['clicks_per_active_day']=total/(ad+1);x['records_per_active_day']=rec/(ad+1);x['active_day_fraction']=(ad/(h+1)).clip(0,1);x['recency_fraction']=(x.days_since_last_activity.fillna(h+1)/(h+1)).clip(0,2)
    x['assessment_information_available']=(x.assessment_weight_due.fillna(0)>0).astype(int);x['score_available']=x.mean_score.notna().astype(int);x['missed_due_assessment']=((x.assessment_weight_due.fillna(0)>0)&(x.assessments_submitted.fillna(0)==0)).astype(int);x['mean_score_safe']=x.mean_score.fillna(0).clip(0,100);x['submission_ratio_safe']=x.submission_ratio.fillna(0).clip(0,1);x['late_submission_ratio_safe']=x.late_submission_ratio.fillna(0).clip(0,1);x['weighted_score_fraction_due_safe']=x.weighted_score_fraction_due.fillna(0).clip(0,1);x['registration_lead_days']=(-x.date_registration.fillna(0)).clip(-200,400);x['presentation_period']=x.code_presentation.str[-1];x['module_period']=x.code_module.astype(str)+'_'+x.presentation_period.astype(str)
    resource=[c for c in x.columns if c.startswith('clicks_') and c!='clicks_total' and not c.endswith('_share')]
    for c in resource:x[c+'_share']=x[c].fillna(0)/(total+1)
    return x,resource


def grouped_split(source,seed):
    g=source.groupby('id_student',as_index=False).target_unsuccessful.max(); tr_ids,temp_ids=train_test_split(g.id_student,test_size=.40,random_state=seed,stratify=g.target_unsuccessful);temp=g[g.id_student.isin(temp_ids)];ca_ids,rem_ids=train_test_split(temp.id_student,test_size=.625,random_state=seed+1,stratify=temp.target_unsuccessful);rem=temp[temp.id_student.isin(rem_ids)];cs_ids,te_ids=train_test_split(rem.id_student,test_size=.60,random_state=seed+2,stratify=rem.target_unsuccessful)
    return source.id_student.isin(tr_ids).to_numpy(),source.id_student.isin(ca_ids).to_numpy(),source.id_student.isin(cs_ids).to_numpy(),source.id_student.isin(te_ids).to_numpy()


def pp(num,cat):
    return ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',min_frequency=3,dtype=np.float32))]),cat)],sparse_threshold=.3)


def run_pipeline(source,target,num,cat,seed,pair_id,estimand,representation,horizon,repeat,save_predictions=False):
    overlap=set(source.id_student.unique())&set(target.id_student.unique()); target=target[~target.id_student.isin(overlap)].copy()
    tr,ca,cs,te=grouped_split(source,seed);cols=num+cat;prep=pp(num,cat);Xtr=prep.fit_transform(source.loc[tr,cols]);Xca=prep.transform(source.loc[ca,cols]);Xcs=prep.transform(source.loc[cs,cols]);Xid=prep.transform(source.loc[te,cols]);Xood=prep.transform(target[cols]);y=source.target_unsuccessful.to_numpy(int);yo=target.target_unsuccessful.to_numpy(int)
    model=LogisticRegression(C=1.0,solver='liblinear',max_iter=1000,tol=1e-6);model.fit(Xtr,y[tr]);pca=model.predict_proba(Xca)[:,1];pcs0=model.predict_proba(Xcs)[:,1];pid0=model.predict_proba(Xid)[:,1];po0=model.predict_proba(Xood)[:,1];cal=fit_calibrators(y[ca],pca)['platt'];pcs=np.clip(cal.predict(pcs0),EPS,1-EPS);pid=np.clip(cal.predict(pid0),EPS,1-EPS);po=np.clip(cal.predict(po0),EPS,1-EPS)
    metric_rows=[];selection_rows=[];conformal_rows=[];curve_rows=[]
    base_common={'analysis_id':pair_id,'estimand':estimand,'representation':representation,'horizon_day':horizon,'repeat':repeat,'seed':seed,'model':'logistic_regression','mapping':'platt','overlap_students_removed':len(overlap),'model_iterations':int(model.n_iter_.max()),'calibrator_iterations':getattr(cal,'n_iter_',0),'n_train':int(tr.sum()),'n_probability_calibration':int(ca.sum()),'n_selection_conformal_calibration':int(cs.sum()),'source_total_n':len(source),'target_total_n_after_overlap':len(target)}
    for domain,yy,pred,ids in [('id_test',y[te],pid,source.loc[te,'id_student'].to_numpy()),('ood_test',yo,po,target.id_student.to_numpy())]:
        metric_rows.append({**base_common,'domain':domain,**metrics(yy,pred)})
        curve,_,_=risk_selection_curve(yy,pred);curve['analysis_id']=pair_id;curve['estimand']=estimand;curve['representation']=representation;curve['horizon_day']=horizon;curve['repeat']=repeat;curve['domain']=domain;curve_rows.append(curve)
    confcal=np.maximum(pcs,1-pcs)
    for desired in SELECTION_RATES:
        th=selection_threshold(confcal,desired)
        for domain,yy,pred in [('id_test',y[te],pid),('ood_test',yo,po)]:selection_rows.append({**base_common,'domain':domain,'desired_selection_rate':desired,'source_threshold':th,**selective_metrics(yy,pred,th)})
    for alpha in ALPHAS:
        qm,k,n=marginal_qhat(y[cs],pcs,alpha);ql=label_conditional_qhat(y[cs],pcs,alpha)
        for method,qobj in [('marginal',qm),('label_conditional',ql)]:
            for domain,yy,pred in [('id_test',y[te],pid),('ood_test',yo,po)]:conformal_rows.append({**base_common,'domain':domain,'alpha':alpha,'nominal_coverage':1-alpha,'conformal_method':method,'marginal_rank_k':k,'marginal_calibration_n':n,'class0_rank_k':ql[0]['k'],'class0_calibration_n':ql[0]['n'],'class1_rank_k':ql[1]['k'],'class1_calibration_n':ql[1]['n'],**conformal_metrics(yy,pred,qobj,method=method)})
    pred_df=None
    if save_predictions:
        # Store primary alpha=.10 marginal and label-conditional membership on target.
        qm,_,_=marginal_qhat(y[cs],pcs,.10);ql=label_conditional_qhat(y[cs],pcs,.10)
        def memberships(qobj,method):
            if method=='marginal':q0=q1=qobj
            else:q0=qobj[0]['qhat'];q1=qobj[1]['qhat']
            i0=po<=q0;i1=(1-po)<=q1;return i0,i1
        m0,m1=memberships(qm,'marginal');l0,l1=memberships(ql,'label_conditional')
        pred_df=pd.DataFrame({'analysis_id':pair_id,'estimand':estimand,'representation':representation,'horizon_day':horizon,'repeat':repeat,'id_student':target.id_student.to_numpy(),'row_key':target.row_key.to_numpy(),'y':yo,'p':po,'marginal_include0':m0,'marginal_include1':m1,'label_conditional_include0':l0,'label_conditional_include1':l1})
    split_df=pd.DataFrame({'analysis_id':pair_id,'estimand':estimand,'representation':representation,'horizon_day':horizon,'repeat':repeat,'row_key':source.row_key,'id_student':source.id_student,'partition':np.where(tr,'train',np.where(ca,'probability_calibration',np.where(cs,'selection_conformal_calibration',np.where(te,'id_test',''))))})
    return metric_rows,selection_rows,conformal_rows,curve_rows,pred_df,split_df


def matched_pairs(df):
    out=[]
    for m in sorted(df.code_module.unique()):
        for p in ['B','J']:
            if ((df.code_module==m)&(df.code_presentation==f'2013{p}')).any() and ((df.code_module==m)&(df.code_presentation==f'2014{p}')).any():out.append((m,p))
    return out


def bootstrap_two_stage(preds,n_boot=500,seed=BASE+8000):
    rng=np.random.default_rng(seed);rows=[]
    for horizon,g_h in preds.groupby('horizon_day'):
        pairs=sorted(g_h.analysis_id.unique())
        for b in range(n_boot):
            chosen=rng.choice(pairs,size=len(pairs),replace=True);pair_values=[]
            for j,pair in enumerate(chosen):
                g=g_h[g_h.analysis_id==pair];students=g.id_student.unique();ss=rng.choice(students,size=len(students),replace=True);parts=[]
                for sid in ss:parts.append(g[g.id_student==sid].sample(n=1,replace=True,random_state=int(rng.integers(1,2**31-1))))
                z=pd.concat(parts,ignore_index=True);y=z.y.to_numpy(int);p=z.p.to_numpy(float);mm=z.marginal_include0.to_numpy(bool);m1=z.marginal_include1.to_numpy(bool);covered=np.where(y==1,m1,mm);mv=metrics(y,p)
                pair_values.append({'n':len(z),'auroc':mv['auroc'],'ece':mv['ece'],'coverage':covered.mean()})
            for weighting in ['equal_pair','size_weighted']:
                weights=np.ones(len(pair_values)) if weighting=='equal_pair' else np.array([x['n'] for x in pair_values],float)
                for met in ['auroc','ece','coverage']:
                    vals=np.array([x[met] for x in pair_values],float);rows.append({'bootstrap':b,'horizon_day':horizon,'weighting':weighting,'metric':met,'value':np.average(vals,weights=weights)})
    return pd.DataFrame(rows)


def main():
    t0=time.time();df,resource=add_features(pd.read_csv(DATA));pairs=matched_pairs(df);pd.DataFrame(pairs,columns=['code_module','presentation_period']).to_csv(MAN/'oulad_matched_pairs.csv',index=False)
    behavior=['num_of_prev_attempts','studied_credits','registration_lead_days','log_clicks_total','log_vle_record_count','clicks_per_active_day','records_per_active_day','active_day_fraction','recency_fraction','no_vle_activity']+[c+'_share' for c in resource]
    assessment=behavior+['assessment_information_available','score_available','missed_due_assessment','mean_score_safe','submission_ratio_safe','late_submission_ratio_safe','weighted_score_fraction_due_safe']
    metric_rows=[];sel_rows=[];conf_rows=[];curve_rows=[];pred_rows=[];split_rows=[];pair_counts=[]
    # Pair-specific primary landmark analysis.
    for h in HORIZONS:
        dh=df[(df.horizon_day==h)&(df.registered_by_horizon==1)].copy();land=dh[dh.withdrawn_by_horizon==0].copy()
        for m,p in pairs:
            src=land[(land.code_module==m)&(land.code_presentation==f'2013{p}')].copy();tgt=land[(land.code_module==m)&(land.code_presentation==f'2014{p}')].copy();overlap=set(src.id_student)&set(tgt.id_student);tgt_eff=tgt[~tgt.id_student.isin(overlap)]
            pair_counts.append({'horizon_day':h,'pair':f'{m}-{p}','source_n':len(src),'source_prevalence':src.target_unsuccessful.mean(),'target_n_before_overlap':len(tgt),'target_n':len(tgt_eff),'target_prevalence':tgt_eff.target_unsuccessful.mean(),'overlap_students_removed':len(overlap)})
            for rep in range(NREP):
                seed=BASE+1009*rep+h+sum(map(ord,m))+ord(p);r=run_pipeline(src,tgt,behavior,BASE_CAT,seed,f'{m}-{p}','landmark_still_registered','score_free',h,rep,save_predictions=(rep==0));metric_rows+=r[0];sel_rows+=r[1];conf_rows+=r[2];curve_rows+=r[3];
                if r[4] is not None:pred_rows.append(r[4])
                split_rows.append(r[5])
            print('OULAD pair',h,m,p,'done',flush=True)
    # Pooled fixed-effect estimand and representation sensitivities.
    for h in HORIZONS:
        dh=df[(df.horizon_day==h)&(df.registered_by_horizon==1)].copy()
        for estimand,cohort in [('landmark_still_registered',dh[dh.withdrawn_by_horizon==0].copy()),('all_registered_by_horizon',dh.copy())]:
            src=pd.concat([cohort[(cohort.code_module==m)&(cohort.code_presentation==f'2013{p}')] for m,p in pairs],ignore_index=True);tgt=pd.concat([cohort[(cohort.code_module==m)&(cohort.code_presentation==f'2014{p}')] for m,p in pairs],ignore_index=True)
            reps=[('score_free',behavior)]
            if h==56 and estimand=='landmark_still_registered':reps.append(('assessment_score_inclusive',assessment))
            for representation,num in reps:
                for rep in range(NREP):
                    seed=BASE+1009*rep+h+73+(0 if estimand.startswith('landmark') else 5000);r=run_pipeline(src,tgt,num,BASE_CAT+['code_module','presentation_period'],seed,'pooled_fixed_effects',estimand,representation,h,rep,save_predictions=False);metric_rows+=r[0];sel_rows+=r[1];conf_rows+=r[2];curve_rows+=r[3];split_rows.append(r[5])
                print('OULAD pooled',h,estimand,representation,'done',flush=True)
    met=pd.DataFrame(metric_rows);sel=pd.DataFrame(sel_rows);conf=pd.DataFrame(conf_rows);curves=pd.concat(curve_rows,ignore_index=True);preds=pd.concat(pred_rows,ignore_index=True);splits=pd.concat(split_rows,ignore_index=True);counts=pd.DataFrame(pair_counts)
    met.to_csv(OUT/'oulad_metrics_complete.csv.gz',index=False,compression='gzip');sel.to_csv(OUT/'oulad_selection_complete.csv.gz',index=False,compression='gzip');conf.to_csv(OUT/'oulad_conformal_complete.csv.gz',index=False,compression='gzip');curves.to_csv(OUT/'oulad_risk_selection_curves.csv.gz',index=False,compression='gzip');preds.to_csv(OUT/'oulad_primary_pair_predictions.csv.gz',index=False,compression='gzip');splits.to_csv(MAN/'oulad_split_manifest.csv.gz',index=False,compression='gzip');counts.to_csv(OUT/'oulad_pair_counts_prevalence.csv',index=False)
    metric_cols=['prevalence','auroc','average_precision','pr_skill','brier','log_loss','ece','calibration_intercept','calibration_slope','hcep_080','hcep_090','hcep_095','aurc','excess_aurc']
    empirical_summary(met,['analysis_id','estimand','representation','horizon_day','domain'],metric_cols).to_csv(OUT/'oulad_metrics_summary.csv',index=False);empirical_summary(sel,['analysis_id','estimand','representation','horizon_day','domain','desired_selection_rate'],['selection_rate','selective_risk','selective_accuracy']).to_csv(OUT/'oulad_selection_summary.csv',index=False);empirical_summary(conf,['analysis_id','estimand','representation','horizon_day','domain','alpha','conformal_method'],['coverage','avg_set_size','singleton_rate','ambiguous_rate','empty_rate','singleton_accuracy','class0_coverage','class1_coverage','class0_avg_set_size','class1_avg_set_size','class0_singleton_accuracy','class1_singleton_accuracy']).to_csv(OUT/'oulad_conformal_summary.csv',index=False)
    # Macro summaries from pair-specific results, equal and target-n weighted.
    pair_met=met[met.analysis_id!='pooled_fixed_effects'].copy();weight_map=counts.set_index(['horizon_day','pair'])[['source_n','target_n']].to_dict('index');macro=[]
    for keys,g in pair_met.groupby(['horizon_day','repeat','domain']):
        h,rep,domain=keys; ws=np.array([weight_map[(h,a)]['source_n' if domain=='id_test' else 'target_n'] for a in g.analysis_id],float)
        for agg,w in [('equal_pair',np.ones(len(g))),('size_weighted',ws)]:
            row={'horizon_day':h,'repeat':rep,'domain':domain,'aggregation':agg}
            for c in metric_cols:row[c]=np.average(g[c],weights=w)
            macro.append(row)
    macro=pd.DataFrame(macro);macro.to_csv(OUT/'oulad_pair_macro_by_repeat.csv',index=False);empirical_summary(macro,['horizon_day','domain','aggregation'],metric_cols).to_csv(OUT/'oulad_pair_macro_summary.csv',index=False)
    # Conformal macros separately.
    pc=conf[(conf.analysis_id!='pooled_fixed_effects')&(conf.alpha==.10)].copy();cmacro=[]
    for keys,g in pc.groupby(['horizon_day','repeat','domain','conformal_method']):
        h,rep,domain,method=keys;ws=np.array([weight_map[(h,a)]['source_n' if domain=='id_test' else 'target_n'] for a in g.analysis_id],float)
        for agg,w in [('equal_pair',np.ones(len(g))),('size_weighted',ws)]:
            row={'horizon_day':h,'repeat':rep,'domain':domain,'conformal_method':method,'aggregation':agg}
            for c in ['coverage','avg_set_size','singleton_rate','class0_coverage','class1_coverage','class0_avg_set_size','class1_avg_set_size','class0_singleton_accuracy','class1_singleton_accuracy']:row[c]=np.average(g[c],weights=w)
            cmacro.append(row)
    cmacro=pd.DataFrame(cmacro);cmacro.to_csv(OUT/'oulad_conformal_macro_by_repeat.csv',index=False);empirical_summary(cmacro,['horizon_day','domain','conformal_method','aggregation'],['coverage','avg_set_size','singleton_rate','class0_coverage','class1_coverage','class0_avg_set_size','class1_avg_set_size','class0_singleton_accuracy','class1_singleton_accuracy']).to_csv(OUT/'oulad_conformal_macro_summary.csv',index=False)
    # Module-period composition.
    mix=[];mixsum=[]
    for h in HORIZONS:
        dh=df[(df.horizon_day==h)&(df.registered_by_horizon==1)&(df.withdrawn_by_horizon==0)];src=pd.concat([dh[(dh.code_module==m)&(dh.code_presentation==f'2013{p}')] for m,p in pairs]);tgt=pd.concat([dh[(dh.code_module==m)&(dh.code_presentation==f'2014{p}')] for m,p in pairs]);ps=np.array([((src.code_module==m)&(src.presentation_period==p)).mean() for m,p in pairs]);pt=np.array([((tgt.code_module==m)&(tgt.presentation_period==p)).mean() for m,p in pairs]);
        for (m,p),a,b in zip(pairs,ps,pt):mix.append({'horizon_day':h,'pair':f'{m}-{p}','source_share':a,'target_share':b,'absolute_difference':abs(a-b)})
        mixsum.append({'horizon_day':h,'total_variation_distance':.5*np.abs(ps-pt).sum(),'jensen_shannon_distance':float(jensenshannon(ps,pt,base=2)),'source_n':len(src),'target_n':len(tgt)})
    pd.DataFrame(mix).to_csv(OUT/'oulad_module_mix.csv',index=False);pd.DataFrame(mixsum).to_csv(OUT/'oulad_module_mix_summary.csv',index=False)
    boot=bootstrap_two_stage(preds,500);boot.to_csv(OUT/'oulad_two_stage_bootstrap.csv.gz',index=False,compression='gzip');empirical_summary(boot,['horizon_day','weighting','metric'],['value']).to_csv(OUT/'oulad_two_stage_bootstrap_summary.csv',index=False)
    save_environment(OUT/'oulad_environment.json');(OUT/'oulad_metadata.json').write_text(json.dumps({'base_seed':BASE,'repeats':NREP,'horizons':HORIZONS,'matched_pairs':[f'{m}-{p}' for m,p in pairs],'primary_estimand':'eventual unsuccessful completion among students registered and not yet withdrawn at the landmark day','secondary_estimand':'eventual unsuccessful completion among all students registered by the landmark day, including early withdrawals','runtime_seconds':time.time()-t0},indent=2),encoding='utf-8')
    print('OULAD ANALYSIS COMPLETE',time.time()-t0,flush=True)

if __name__=='__main__':main()
