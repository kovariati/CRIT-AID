import os
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(os.environ.get('CRIT_AID_ROOT', Path(__file__).resolve().parents[1]));O=ROOT/'outputs';D=ROOT/'docs';F=ROOT/'figures'
plt.rcParams.update({'font.size':12,'axes.titlesize':13,'axes.labelsize':12,'legend.fontsize':10,'xtick.labelsize':10,'ytick.labelsize':10})
m=pd.read_csv(O/'acs_metrics_complete.csv.gz');p=m[(m.model=='logistic_regression')&(m.mapping=='platt')&(m.weighting=='survey_weighted')]
rows=[]
for (target,domain),g in p.groupby(['target_definition','domain']):
 z={'target_definition':target,'domain':domain}
 for met in ['prevalence','auroc','average_precision','pr_skill','brier','log_loss','ece','calibration_intercept','calibration_slope','hcep_090','aurc','excess_aurc']:
  z[f'{met}_mean']=g[met].mean();z[f'{met}_sd']=g[met].std(ddof=1)
 rows.append(z)
pd.DataFrame(rows).to_csv(D/'table_acs_primary.csv',index=False)
# paired delta
wide=p.pivot_table(index=['target_definition','repeat'],columns='domain',values=['auroc','average_precision','brier','log_loss','ece','calibration_intercept','calibration_slope','hcep_090','aurc'])
d=[]
for (target,rep),r in wide.iterrows():
 z={'target_definition':target,'repeat':rep}
 for met in ['auroc','average_precision','brier','log_loss','ece','calibration_intercept','calibration_slope','hcep_090','aurc']:z[met+'_delta']=r[(met,'ood_test')]-r[(met,'id_test')]
 d.append(z)
pd.DataFrame(d).to_csv(D/'table_acs_paired_deltas.csv',index=False)
# conformal and selection
c=pd.read_csv(O/'acs_conformal_complete.csv.gz');c=c[(c.model=='logistic_regression')&(c.mapping=='platt')&(c.alpha==.10)]
c.groupby(['target_definition','domain','weighting','conformal_method'])[['coverage','class0_coverage','class1_coverage','avg_set_size','singleton_rate','singleton_accuracy','empty_rate']].agg(['mean','std']).to_csv(D/'table_acs_conformal.csv')
s=pd.read_csv(O/'acs_selection_complete.csv.gz');s=s[(s.model=='logistic_regression')&(s.mapping=='platt')&(s.weighting=='survey_weighted')]
s.groupby(['target_definition','domain','desired_selection_rate'])[['selection_rate','selective_risk']].agg(['mean','std']).to_csv(D/'table_acs_selection.csv')
# target definition metrics figure
target_order=['unadjusted_reported_income_50000','survey_year_adjusted_50000','cpi_constant_2018usd'];labels=['Unadjusted reported','Survey-year adjusted','CPI-constant 2018 USD']
z=p[p.domain=='ood_test'].groupby('target_definition')[['auroc','brier','log_loss','ece']].agg(['mean','std']).reindex(target_order)
fig,axes=plt.subplots(1,4,figsize=(14,4.3))
for ax,met,title in zip(axes,['auroc','brier','log_loss','ece'],['AUROC','Brier score','Log loss','ECE']):
 vals=z[(met,'mean')].to_numpy();errs=z[(met,'std')].to_numpy();x=np.arange(3);ax.bar(x,vals,yerr=errs,capsize=3);ax.set_xticks(x);ax.set_xticklabels(labels,rotation=35,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.3)
fig.tight_layout();fig.savefig(F/'figure_acs_target_metrics.png',dpi=300);plt.close(fig)
# target definition paired bootstrap plot
b=pd.read_csv(O/'acs_target_definition_paired_bootstrap.csv.gz')
fig,axes=plt.subplots(1,4,figsize=(13,4))
for ax,met in zip(axes,['auroc','brier','log_loss','ece']):
 v=b[b.metric==met].delta;ax.boxplot(v,showfliers=False);ax.axhline(0,linestyle='--',linewidth=1);ax.set_xticks([]);ax.set_title(f'Unadjusted − CPI\n{met.replace("_"," ")}');ax.grid(axis='y',alpha=.3)
fig.tight_layout();fig.savefig(F/'figure_acs_paired_bootstrap.png',dpi=300);plt.close(fig)
# reliability diagram
rb=pd.read_csv(O/'acs_primary_reliability_bins.csv');fig,ax=plt.subplots(figsize=(6.5,5.5));ax.plot([0,1],[0,1],linestyle='--',linewidth=1);ax.plot(rb.mean_probability,rb.observed_rate,'o-');ax.set_xlabel('Mean predicted probability');ax.set_ylabel('Observed weighted event rate');ax.set_title('ACS CPI-constant target, fixed primary refit');ax.grid(alpha=.3);fig.tight_layout();fig.savefig(F/'figure_acs_reliability.png',dpi=300);plt.close(fig)
# subgroup ECE
sg=pd.read_csv(O/'acs_primary_subgroup_metrics.csv');sg=sg.sort_values(['group_type','ece']);fig,ax=plt.subplots(figsize=(9,6));names=(sg.group_type+': '+sg.group).tolist();ax.barh(np.arange(len(sg)),sg.ece);ax.set_yticks(np.arange(len(sg)));ax.set_yticklabels(names);ax.set_xlabel('Survey-weighted ECE');ax.set_title('ACS CPI-constant target subgroup diagnostics');ax.grid(axis='x',alpha=.3);fig.tight_layout();fig.savefig(F/'figure_acs_subgroups.png',dpi=300);plt.close(fig)
# independent cohort sensitivity
ic=pd.read_csv(O/'acs_independent_cohort_sensitivity.csv');ic.to_csv(D/'table_acs_independent_cohorts.csv',index=False)
print('ACS tables figures built')
