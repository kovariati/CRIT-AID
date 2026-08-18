import os
from pathlib import Path
import json
import numpy as np,pandas as pd
ROOT=Path(os.environ.get('CRIT_AID_ROOT', Path(__file__).resolve().parents[1]));O=ROOT/'outputs';D=ROOT/'docs';D.mkdir(exist_ok=True)
R={}
def summary(g,cols):
 return {c:{'mean':float(g[c].mean()),'sd':float(g[c].std(ddof=1))} for c in cols}
# ACS primary
m=pd.read_csv(O/'acs_metrics_complete.csv.gz');p=m[(m.model=='logistic_regression')&(m.mapping=='platt')&(m.weighting=='survey_weighted')]
for target in p.target_definition.unique():
 R.setdefault('acs',{})[target]={}
 for dom in ['id_test','ood_test']:
  R['acs'][target][dom]=summary(p[(p.target_definition==target)&(p.domain==dom)],['prevalence','auroc','average_precision','pr_skill','brier','log_loss','ece','calibration_intercept','calibration_slope','hcep_090','aurc','excess_aurc'])
# ACS selection and conformal
s=pd.read_csv(O/'acs_selection_complete.csv.gz');s=s[(s.model=='logistic_regression')&(s.mapping=='platt')&(s.weighting=='survey_weighted')&(s.desired_selection_rate==.8)]
c=pd.read_csv(O/'acs_conformal_complete.csv.gz');c=c[(c.model=='logistic_regression')&(c.mapping=='platt')&(c.weighting=='survey_weighted')&(c.alpha==.1)]
for target in R['acs']:
 for dom in ['id_test','ood_test']:
  R['acs'][target][dom]['selection_080']=summary(s[(s.target_definition==target)&(s.domain==dom)],['selection_rate','selective_risk'])
  for method in ['marginal','label_conditional']:
   R['acs'][target][dom][f'conformal_{method}']=summary(c[(c.target_definition==target)&(c.domain==dom)&(c.conformal_method==method)],['coverage','class0_coverage','class1_coverage','avg_set_size','singleton_rate','singleton_accuracy'])
# bootstrap
b=pd.read_csv(O/'acs_target_definition_paired_bootstrap.csv.gz')
R['acs']['paired_bootstrap']={}
for met,g in b.groupby('metric'):R['acs']['paired_bootstrap'][met]={'mean':float(g.delta.mean()),'q025':float(g.delta.quantile(.025)),'q975':float(g.delta.quantile(.975))}
# independent cohorts
ic=pd.read_csv(O/'acs_independent_cohort_sensitivity.csv')
R['acs']['independent_cohorts']={}
for (target,dom),g in ic.groupby(['target_definition','domain']):R['acs']['independent_cohorts'].setdefault(target,{})[dom]=summary(g,['auroc','brier','log_loss','ece'])
# OULAD macro
om=pd.read_csv(O/'oulad_pair_macro_by_repeat.csv');R['oulad']={}
for h in [14,56]:
 R['oulad'][str(h)]={}
 for agg in ['equal_pair','size_weighted']:
  R['oulad'][str(h)][agg]={}
  for dom in ['id_test','ood_test']:
   R['oulad'][str(h)][agg][dom]=summary(om[(om.horizon_day==h)&(om.aggregation==agg)&(om.domain==dom)],['prevalence','auroc','average_precision','pr_skill','brier','log_loss','ece','hcep_090','aurc','excess_aurc'])
oc=pd.read_csv(O/'oulad_conformal_macro_by_repeat.csv')
for h in [14,56]:
 for agg in ['equal_pair','size_weighted']:
  for dom in ['id_test','ood_test']:
   for method in ['marginal','label_conditional']:
    R['oulad'][str(h)][agg][dom][f'conformal_{method}']=summary(oc[(oc.horizon_day==h)&(oc.aggregation==agg)&(oc.domain==dom)&(oc.conformal_method==method)],['coverage','class0_coverage','class1_coverage','avg_set_size','singleton_rate','class0_singleton_accuracy','class1_singleton_accuracy'])
# OULAD pooled estimands
op=pd.read_csv(O/'oulad_metrics_complete.csv.gz');op=op[op.analysis_id=='pooled_fixed_effects'];R['oulad']['pooled']={}
for keys,g in op.groupby(['horizon_day','estimand','representation','domain']):
 h,e,r,d=keys;R['oulad']['pooled'].setdefault(str(h),{}).setdefault(e,{}).setdefault(r,{})[d]=summary(g,['prevalence','auroc','average_precision','brier','log_loss','ece'])
# OULAD bootstrap
ob=pd.read_csv(O/'oulad_two_stage_bootstrap.csv.gz');R['oulad']['bootstrap']={}
for keys,g in ob.groupby(['horizon_day','weighting','metric']):
 h,w,met=keys;R['oulad']['bootstrap'].setdefault(str(h),{}).setdefault(w,{})[met]={'mean':float(g.value.mean()),'q025':float(g.value.quantile(.025)),'q975':float(g.value.quantile(.975))}
# South
sm=pd.read_csv(O/'south_metrics_complete.csv.gz');sm=sm[sm.mapping=='platt'].copy();sm['group']=np.where(sm.scenario.str.startswith('random_three_features_draw_'),'random_three_features',sm.scenario)
sc=pd.read_csv(O/'south_conformal_complete.csv.gz');sc=sc[(sc.mapping=='platt')&(sc.alpha==.1)].copy();sc['group']=np.where(sc.scenario.str.startswith('random_three_features_draw_'),'random_three_features',sc.scenario)
R['south']={}
for group,g in sm.groupby('group'):
 R['south'][group]=summary(g,['auroc','average_precision','pr_skill','brier','log_loss','ece','hcep_090','aurc','excess_aurc'])
 for method in ['marginal','label_conditional']:
  R['south'][group][f'conformal_{method}']=summary(sc[(sc.group==group)&(sc.conformal_method==method)],['coverage','class0_coverage','class1_coverage','avg_set_size','singleton_rate','singleton_accuracy'])
cont=pd.read_csv(O/'south_targeted_vs_random_distribution.csv');R['south']['targeted_distribution']={}
for met,g in cont.groupby('metric'):R['south']['targeted_distribution'][met]={'targeted_minus_random_median_mean':float(g.targeted_minus_random_median.mean()),'severity_percentile_mean':float(g.targeted_severity_percentile_among_random.mean()),'severity_percentile_median':float(g.targeted_severity_percentile_among_random.median())}
# Heart
hm=pd.read_csv(O/'heart_metrics_complete.csv.gz');hm=hm[(hm.mapping=='platt')&(~hm.missing_indicators)];hc=pd.read_csv(O/'heart_conformal_complete.csv.gz');hc=hc[(hc.mapping=='platt')&(~hc.missing_indicators)&(hc.alpha==.1)];hs=pd.read_csv(O/'heart_selection_complete.csv.gz');hs=hs[(hs.mapping=='platt')&(~hs.missing_indicators)&(hs.desired_selection_rate==.8)]
R['heart']={}
for site in hm.heldout_site.unique():
 R['heart'][site]={}
 for dom in ['id_test','ood_test']:
  R['heart'][site][dom]=summary(hm[(hm.heldout_site==site)&(hm.domain==dom)],['prevalence','auroc','average_precision','pr_skill','brier','log_loss','ece','hcep_090','aurc','excess_aurc'])
  R['heart'][site][dom]['selection_080']=summary(hs[(hs.heldout_site==site)&(hs.domain==dom)],['selection_rate','selective_risk'])
  for method in ['marginal','label_conditional']:
   R['heart'][site][dom][f'conformal_{method}']=summary(hc[(hc.heldout_site==site)&(hc.domain==dom)&(hc.conformal_method==method)],['coverage','class0_coverage','class1_coverage','avg_set_size','singleton_rate','singleton_accuracy'])
# provenance
R['heart']['site_counts']=pd.read_csv(O/'heart_site_counts_prevalence.csv').to_dict('records')
(D/'result_registry.json').write_text(json.dumps(R,indent=2),encoding='utf-8')
# Flatten for audit
flat=[]
def rec(path,obj):
 if isinstance(obj,dict):
  for k,v in obj.items():rec(path+[str(k)],v)
 elif isinstance(obj,list):
  flat.append({'key':'/'.join(path),'value_json':json.dumps(obj)})
 else:flat.append({'key':'/'.join(path),'value':obj})
rec([],R);pd.DataFrame(flat).to_csv(D/'result_registry_flat.csv',index=False)
print('registry built')
