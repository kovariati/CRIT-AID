from __future__ import annotations

import os
import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    EPS, fit_calibrators, metrics, selection_threshold, selective_metrics,
    risk_selection_curve, marginal_qhat, label_conditional_qhat,
    conformal_metrics, empirical_summary, clopper_pearson,
)

ROOT=Path(os.environ.get('CRIT_AID_ROOT', Path(__file__).resolve().parents[1])); PREP=ROOT/'prepared'; OUT=ROOT/'outputs'; MAN=ROOT/'manifests'
OUT.mkdir(exist_ok=True); MAN.mkdir(exist_ok=True)
BASE=20260728
SELECTION_RATES=[0.50,0.70,0.80,0.90,0.95]
ALPHAS=[0.05,0.10,0.20]


def split4(y, seed, strata=None):
    y=np.asarray(y,int); idx=np.arange(len(y)); strat=y if strata is None else np.asarray(strata)
    tr,temp=train_test_split(idx,test_size=.40,random_state=seed,stratify=strat)
    strat_temp=y[temp] if strata is None else np.asarray(strata)[temp]
    ca,rem=train_test_split(temp,test_size=.625,random_state=seed+1,stratify=strat_temp)
    strat_rem=y[rem] if strata is None else np.asarray(strata)[rem]
    cs,te=train_test_split(rem,test_size=.60,random_state=seed+2,stratify=strat_rem)
    return tr,ca,cs,te


def preprocessor(num,cat,indicators=False):
    return ColumnTransformer([
        ('num',Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=indicators)),('sc',StandardScaler())]),num),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent',add_indicator=indicators)),('oh',OneHotEncoder(handle_unknown='ignore',min_frequency=2,dtype=np.float32))]),cat),
    ],sparse_threshold=.3)


def base_model():
    return LogisticRegression(C=1.0,solver='liblinear',max_iter=2000,tol=1e-7,random_state=BASE)

# SOUTH GERMAN CREDIT
SNUM=['laufzeit','hoehe','alter']
SCAT=['laufkont','moral','verw','sparkont','beszeit','rate','famges','buerge','wohnzeit','verm','weitkred','wohn','bishkred','beruf','pers','telef','gastarb']
SFEAT=SNUM+SCAT
SREADABLE={
'laufkont':'checking-account status','laufzeit':'duration','moral':'credit history','verw':'purpose','hoehe':'credit amount','sparkont':'savings status','beszeit':'employment duration','rate':'installment rate','famges':'personal status and sex','buerge':'other debtors or guarantors','wohnzeit':'present residence duration','verm':'property','alter':'age','weitkred':'other installment plans','wohn':'housing','bishkred':'number of existing credits','beruf':'job','pers':'people liable','telef':'telephone','gastarb':'foreign-worker status'}


def rank_south_features(train, y, seed):
    z=pd.DataFrame(index=train.index); discrete=[]
    for c in SFEAT:
        if c in SCAT:
            z[c]=pd.factorize(train[c].astype(str),sort=True)[0];discrete.append(True)
        else:
            v=pd.to_numeric(train[c],errors='coerce');z[c]=v.fillna(v.median());discrete.append(False)
    mi=mutual_info_classif(z[SFEAT],y,discrete_features=discrete,random_state=seed)
    order=np.argsort(-mi,kind='mergesort')
    return [SFEAT[i] for i in order], dict(zip(SFEAT,mi))


def fixed_three_per_record(x,rng):
    z=x.copy(); cols=np.asarray(SFEAT)
    for pos,idx in enumerate(z.index):
        chosen=rng.choice(cols,3,replace=False)
        z.loc[idx,chosen]=np.nan
    return z


def measurement_noise(x,train,rng,rate=.20):
    z=x.copy()
    for c in SNUM:
        vals=pd.to_numeric(z[c],errors='coerce').to_numpy(float); sd=float(pd.to_numeric(train[c],errors='coerce').std())
        mask=rng.random(len(z))<rate; vals[mask]+=rng.normal(0,.20*sd,mask.sum())
        lo=float(pd.to_numeric(train[c],errors='coerce').min());hi=float(pd.to_numeric(train[c],errors='coerce').max())
        vals=np.clip(vals,lo,hi)
        if c in ['laufzeit','alter']: vals=np.rint(vals)
        z[c]=vals
    for c in SCAT:
        mask=rng.random(len(z))<rate
        freq=train[c].value_counts(normalize=True,dropna=True)
        cats=freq.index.to_numpy(); probs=freq.to_numpy(float)
        ci=z.columns.get_loc(c)
        for i in np.flatnonzero(mask):
            original=z.iloc[i,ci]
            ok=cats!=original
            if ok.sum(): z.iat[i,ci]=rng.choice(cats[ok],p=probs[ok]/probs[ok].sum())
    return z


def run_south():
    d=pd.read_csv(PREP/'south_german_with_ids.csv.gz')
    d['target_bad_credit']=(d['kredit']==0).astype(int)
    for c in SCAT:d[c]=d[c].astype('object')
    y=d.target_bad_credit.to_numpy(int)
    metric_rows=[];sel_rows=[];conf_rows=[];curve_rows=[];rank_rows=[];split_rows=[];conv=[]
    for rep in range(20):
        seed=BASE+131*rep
        tr,ca,cs,te=split4(y,seed)
        assignment=np.full(len(d),'',object);assignment[tr]='train';assignment[ca]='probability_calibration';assignment[cs]='selection_conformal_calibration';assignment[te]='id_test'
        split_rows.append(pd.DataFrame({'row_id':d.row_id,'repeat':rep,'seed':seed,'partition':assignment}))
        train=d.loc[tr].reset_index(drop=True);cald=d.loc[ca].reset_index(drop=True);confd=d.loc[cs].reset_index(drop=True);test=d.loc[te].reset_index(drop=True)
        rank,mi=rank_south_features(train,train.target_bad_credit.to_numpy(int),seed)
        rank_rows += [{'repeat':rep,'rank':i+1,'feature_code':f,'feature_name':SREADABLE[f],'mutual_information':mi[f]} for i,f in enumerate(rank)]
        pp=preprocessor(SNUM,SCAT,False)
        Xtr=pp.fit_transform(train[SFEAT]);Xca=pp.transform(cald[SFEAT]);Xcs=pp.transform(confd[SFEAT])
        m=base_model();m.fit(Xtr,train.target_bad_credit)
        pca0=m.predict_proba(Xca)[:,1];pcs0=m.predict_proba(Xcs)[:,1]
        cals=fit_calibrators(cald.target_bad_credit,pca0)
        rng=np.random.default_rng(seed+10000)
        scenarios=[('clean',test[SFEAT].copy(),'none','')]
        scenarios.append(('fixed_count_mcar_like_three_cells_per_record',fixed_three_per_record(test[SFEAT],rng),'fixed_count_mcar_like',''))
        scenarios.append(('targeted_top3',test[SFEAT].assign(**{f:np.nan for f in rank[:3]}),'targeted_feature_deletion','|'.join(rank[:3])))
        scenarios.append(('measurement_and_category_error',measurement_noise(test[SFEAT],train[SFEAT],rng),'label_independent_measurement_corruption',''))
        # retain every random set separately
        for draw in range(20):
            fs=tuple(sorted(rng.choice(SFEAT,3,replace=False).tolist()))
            z=test[SFEAT].copy();z.loc[:,list(fs)]=np.nan
            scenarios.append((f'random_three_features_draw_{draw:02d}',z,'random_feature_set_deletion','|'.join(fs)))
        # label-informed oracle/adversarial stress, kept separate
        adv=test[SFEAT].copy();yt=test.target_bad_credit.to_numpy(int)
        affected=rng.choice(len(test),size=math.ceil(.30*len(test)),replace=False)
        for i in affected:
            pool=train[train.target_bad_credit != yt[i]]
            donor=pool.iloc[int(rng.integers(len(pool)))]
            for f in rank[:3]:adv.at[i,f]=donor[f]
        scenarios.append(('label_informed_oracle_adversarial',adv,'label_informed_oracle','|'.join(rank[:3])))
        for mapping,cal in cals.items():
            if cal is None:continue
            pcs=np.clip(cal.predict(pcs0),EPS,1-EPS)
            for scenario,x,scenario_family,feature_set in scenarios:
                # stress scenarios are primary Platt only; clean includes all mappings
                if scenario!='clean' and mapping!='platt':continue
                p=np.clip(cal.predict(m.predict_proba(pp.transform(x))[:,1]),EPS,1-EPS)
                base={'repeat':rep,'seed':seed,'model':'logistic_regression','mapping':mapping,'scenario':scenario,'scenario_family':scenario_family,'feature_set':feature_set,'top3_features':'|'.join(rank[:3]),'domain':'test'}
                metric_rows.append({**base,**metrics(yt,p)})
                conf_cal=np.maximum(pcs,1-pcs)
                for desired in SELECTION_RATES:
                    thr=selection_threshold(conf_cal,desired)
                    sel_rows.append({**base,'desired_selection_rate':desired,'source_threshold':thr,**selective_metrics(yt,p,thr)})
                if mapping=='platt':
                    curve,_,_=risk_selection_curve(yt,p);curve=curve.assign(**base);curve_rows.append(curve)
                for alpha in ALPHAS:
                    q,k,n=marginal_qhat(confd.target_bad_credit,pcs,alpha)
                    conf_rows.append({**base,'alpha':alpha,'nominal_coverage':1-alpha,'conformal_method':'marginal','calibration_n':n,'rank_k':k,'rank_fraction':k/(n+1),'qhat0':q,'qhat1':q,**conformal_metrics(yt,p,q,method='marginal')})
                    qlc=label_conditional_qhat(confd.target_bad_credit,pcs,alpha)
                    conf_rows.append({**base,'alpha':alpha,'nominal_coverage':1-alpha,'conformal_method':'label_conditional','calibration_n':n,'rank_k':np.nan,'rank_fraction':np.nan,'qhat0':qlc[0]['qhat'],'qhat1':qlc[1]['qhat'],'class0_calibration_n':qlc[0]['n'],'class1_calibration_n':qlc[1]['n'],'class0_rank_k':qlc[0]['k'],'class1_rank_k':qlc[1]['k'],**conformal_metrics(yt,p,qlc,method='label_conditional')})
        conv.append({'repeat':rep,'model':'logistic_regression','started':1,'retained':1,'excluded':0,'iterations':int(np.max(np.atleast_1d(m.n_iter_))),'converged':bool(np.max(np.atleast_1d(m.n_iter_))<m.max_iter)})
        print('South repeat',rep,'done',flush=True)
    met=pd.DataFrame(metric_rows);sel=pd.DataFrame(sel_rows);conf=pd.DataFrame(conf_rows)
    met.to_csv(OUT/'south_metrics_complete.csv.gz',index=False,compression='gzip');sel.to_csv(OUT/'south_selection_complete.csv.gz',index=False,compression='gzip');conf.to_csv(OUT/'south_conformal_complete.csv.gz',index=False,compression='gzip')
    pd.concat(curve_rows,ignore_index=True).to_csv(OUT/'south_risk_selection_curves.csv.gz',index=False,compression='gzip')
    pd.concat(split_rows,ignore_index=True).to_csv(MAN/'south_split_manifest.csv.gz',index=False,compression='gzip')
    pd.DataFrame(rank_rows).to_csv(OUT/'south_feature_ranking_all_refits.csv',index=False);pd.DataFrame(conv).to_csv(OUT/'south_convergence.csv',index=False)
    # random-set distribution and targeted placement, by repeat and metric
    mprimary=met[(met.mapping=='platt')].copy();mprimary['random_draw']=mprimary.scenario.str.startswith('random_three_features_draw_')
    contrasts=[];setdist=[]
    for rep,g in mprimary.groupby('repeat'):
        rnd=g[g.random_draw]
        tar=g[g.scenario=='targeted_top3'].iloc[0]
        clean=g[g.scenario=='clean'].iloc[0]
        for metric in ['auroc','average_precision','brier','log_loss','ece','hcep_090','aurc','excess_aurc']:
            vals=rnd[metric].to_numpy(float);tv=float(tar[metric]);cv=float(clean[metric])
            smaller_better=metric in ['brier','log_loss','ece','hcep_090','aurc','excess_aurc']
            severity=(vals-cv) if smaller_better else (cv-vals);tsev=(tv-cv) if smaller_better else (cv-tv)
            percentile=float(np.mean(severity<=tsev))
            contrasts.append({'repeat':rep,'metric':metric,'clean':cv,'targeted':tv,'random_mean':vals.mean(),'random_sd':vals.std(ddof=1),'random_min':vals.min(),'random_q05':np.quantile(vals,.05),'random_median':np.median(vals),'random_q95':np.quantile(vals,.95),'random_max':vals.max(),'targeted_minus_random_median':tv-np.median(vals),'targeted_severity_percentile_among_random':percentile})
        for _,r in rnd.iterrows():setdist.append({'repeat':rep,'feature_set':r.feature_set,**{x:r[x] for x in ['auroc','brier','ece','hcep_090','aurc']}})
    pd.DataFrame(contrasts).to_csv(OUT/'south_targeted_vs_random_distribution.csv',index=False)
    pd.DataFrame(setdist).to_csv(OUT/'south_random_feature_set_results.csv',index=False)
    empirical_summary(met,['mapping','scenario_family'],['auroc','average_precision','brier','log_loss','ece','hcep_090','aurc','excess_aurc']).to_csv(OUT/'south_metrics_summary.csv',index=False)
    empirical_summary(conf,['mapping','scenario_family','conformal_method','alpha'],['coverage','avg_set_size','singleton_rate','singleton_accuracy','class0_coverage','class1_coverage','class0_singleton_accuracy','class1_singleton_accuracy','empty_rate']).to_csv(OUT/'south_conformal_summary.csv',index=False)

# HEART DISEASE
HNUM=['age','resting_bp','cholesterol','max_heart_rate','oldpeak','major_vessels']
HCAT=['sex','chest_pain_type','fasting_blood_sugar','resting_ecg','exercise_angina','slope','thal']
HFEAT=HNUM+HCAT
SITES=['Cleveland','Hungary','Switzerland','VA Long Beach']

def run_heart():
    d=pd.read_csv(PREP/'heart_disease_with_ids.csv.gz')
    for c in HCAT:d[c]=d[c].astype('object')
    metric_rows=[];sel_rows=[];conf_rows=[];curve_rows=[];split_rows=[];conv=[];reliability=[]
    for held in SITES:
        src=d[d.site!=held].reset_index(drop=True);tgt=d[d.site==held].reset_index(drop=True)
        ys=src.target_disease_present.to_numpy(int);yt=tgt.target_disease_present.to_numpy(int)
        strata=src.site.astype(str)+'|'+src.target_disease_present.astype(str)
        for rep in range(10):
            seed=BASE+149*rep+SITES.index(held)
            tr,ca,cs,te=split4(ys,seed,strata)
            assignment=np.full(len(src),'',object);assignment[tr]='train';assignment[ca]='probability_calibration';assignment[cs]='selection_conformal_calibration';assignment[te]='id_test'
            split_rows.append(pd.DataFrame({'heldout_site':held,'record_key':src.record_key,'source_site':src.site,'repeat':rep,'seed':seed,'partition':assignment}))
            for indicators in [False,True]:
                pp=preprocessor(HNUM,HCAT,indicators)
                Xtr=pp.fit_transform(src.loc[tr,HFEAT]);Xca=pp.transform(src.loc[ca,HFEAT]);Xcs=pp.transform(src.loc[cs,HFEAT]);Xid=pp.transform(src.loc[te,HFEAT]);Xood=pp.transform(tgt[HFEAT])
                m=base_model();m.fit(Xtr,ys[tr]);pca0=m.predict_proba(Xca)[:,1];pcs0=m.predict_proba(Xcs)[:,1];pid0=m.predict_proba(Xid)[:,1];pood0=m.predict_proba(Xood)[:,1]
                cals=fit_calibrators(ys[ca],pca0)
                for mapping,cal in cals.items():
                    if cal is None:continue
                    pcs=np.clip(cal.predict(pcs0),EPS,1-EPS);pid=np.clip(cal.predict(pid0),EPS,1-EPS);pood=np.clip(cal.predict(pood0),EPS,1-EPS)
                    base0={'heldout_site':held,'repeat':rep,'seed':seed,'model':'logistic_regression','mapping':mapping,'missing_indicators':indicators}
                    for domain,y,p in [('id_test',ys[te],pid),('ood_test',yt,pood)]:
                        base={**base0,'domain':domain};metric_rows.append({**base,**metrics(y,p)})
                        for desired in SELECTION_RATES:
                            thr=selection_threshold(np.maximum(pcs,1-pcs),desired)
                            sel_rows.append({**base,'desired_selection_rate':desired,'source_threshold':thr,**selective_metrics(y,p,thr)})
                        if mapping=='platt' and not indicators:
                            curve,_,_=risk_selection_curve(y,p);curve=curve.assign(**base);curve_rows.append(curve)
                        for alpha in ALPHAS:
                            q,k,n=marginal_qhat(ys[cs],pcs,alpha)
                            cm=conformal_metrics(y,p,q,method='marginal')
                            row={**base,'alpha':alpha,'nominal_coverage':1-alpha,'conformal_method':'marginal','calibration_n':n,'rank_k':k,'rank_fraction':k/(n+1),'qhat0':q,'qhat1':q,**cm}
                            for c in [0,1]:
                                # exact interval based on record counts, unweighted
                                include0=p<=q;include1=(1-p)<=q;covered=np.where(y==1,include1,include0);mask=y==c;lo,hi=clopper_pearson(int(covered[mask].sum()),int(mask.sum()))
                                row[f'class{c}_coverage_cp_low']=lo;row[f'class{c}_coverage_cp_high']=hi
                            conf_rows.append(row)
                            qlc=label_conditional_qhat(ys[cs],pcs,alpha);cm=conformal_metrics(y,p,qlc,method='label_conditional')
                            row={**base,'alpha':alpha,'nominal_coverage':1-alpha,'conformal_method':'label_conditional','calibration_n':len(cs),'rank_k':np.nan,'rank_fraction':np.nan,'qhat0':qlc[0]['qhat'],'qhat1':qlc[1]['qhat'],'class0_calibration_n':qlc[0]['n'],'class1_calibration_n':qlc[1]['n'],'class0_rank_k':qlc[0]['k'],'class1_rank_k':qlc[1]['k'],**cm}
                            include0=p<=qlc[0]['qhat'];include1=(1-p)<=qlc[1]['qhat'];covered=np.where(y==1,include1,include0)
                            for c in [0,1]:
                                mask=y==c;lo,hi=clopper_pearson(int(covered[mask].sum()),int(mask.sum()));row[f'class{c}_coverage_cp_low']=lo;row[f'class{c}_coverage_cp_high']=hi
                            conf_rows.append(row)
                    conv.append({'heldout_site':held,'repeat':rep,'missing_indicators':indicators,'model':'logistic_regression','started':1,'retained':1,'excluded':0,'iterations':int(np.max(np.atleast_1d(m.n_iter_))),'converged':bool(np.max(np.atleast_1d(m.n_iter_))<m.max_iter)})
                # reliability bins for primary Platt, target
                if not indicators and cals.get('platt') is not None:
                    from common import expected_calibration_error
                    p=cals['platt'].predict(pood0);_,bins=expected_calibration_error(yt,p,n_bins=10);bins['heldout_site']=held;bins['repeat']=rep;reliability.append(bins)
            print('Heart',held,'repeat',rep,'done',flush=True)
    met=pd.DataFrame(metric_rows);sel=pd.DataFrame(sel_rows);conf=pd.DataFrame(conf_rows)
    met.to_csv(OUT/'heart_metrics_complete.csv.gz',index=False,compression='gzip');sel.to_csv(OUT/'heart_selection_complete.csv.gz',index=False,compression='gzip');conf.to_csv(OUT/'heart_conformal_complete.csv.gz',index=False,compression='gzip')
    pd.concat(curve_rows,ignore_index=True).to_csv(OUT/'heart_risk_selection_curves.csv.gz',index=False,compression='gzip');pd.concat(split_rows,ignore_index=True).to_csv(MAN/'heart_split_manifest.csv.gz',index=False,compression='gzip')
    pd.concat(reliability,ignore_index=True).to_csv(OUT/'heart_reliability_bins.csv',index=False);pd.DataFrame(conv).to_csv(OUT/'heart_convergence.csv',index=False)
    empirical_summary(met,['heldout_site','mapping','missing_indicators','domain'],['prevalence','auroc','average_precision','pr_skill','brier','log_loss','ece','calibration_intercept','calibration_slope','hcep_090','aurc','excess_aurc']).to_csv(OUT/'heart_metrics_summary.csv',index=False)
    empirical_summary(conf,['heldout_site','mapping','missing_indicators','domain','conformal_method','alpha'],['coverage','avg_set_size','singleton_rate','singleton_accuracy','class0_coverage','class1_coverage','class0_avg_set_size','class1_avg_set_size','class0_singleton_accuracy','class1_singleton_accuracy','empty_rate']).to_csv(OUT/'heart_conformal_summary.csv',index=False)
    # explicit site class counts
    d.groupby('site').target_disease_present.agg(n='size',positive_n='sum',prevalence='mean').assign(negative_n=lambda x:x.n-x.positive_n,auprc_baseline=lambda x:x.prevalence).reset_index().to_csv(OUT/'heart_site_counts_prevalence.csv',index=False)

if __name__=='__main__':
    t=time.time();run_south();run_heart();(OUT/'external_analysis_metadata.json').write_text(json.dumps({'runtime_seconds':time.time()-t},indent=2));print('external analysis complete',flush=True)
