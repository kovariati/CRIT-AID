import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(os.environ.get('CRIT_AID_ROOT', Path(__file__).resolve().parents[1]));O=ROOT/'outputs';F=ROOT/'figures';D=ROOT/'docs';F.mkdir(exist_ok=True);D.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.titlesize':13,'axes.labelsize':12,'legend.fontsize':10,'xtick.labelsize':10,'ytick.labelsize':10})

def ms(x): return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"
# OULAD pair-level primary target results
m=pd.read_csv(O/'oulad_metrics_complete.csv.gz');c=pd.read_csv(O/'oulad_conformal_complete.csv.gz')
pm=m[(m.analysis_id!='pooled_fixed_effects')&(m.domain=='ood_test')]
pc=c[(c.analysis_id!='pooled_fixed_effects')&(c.domain=='ood_test')&(c.alpha==.10)]
counts=pd.read_csv(O/'oulad_pair_counts_prevalence.csv')
rows=[]
for (h,pair),g in pm.groupby(['horizon_day','analysis_id']):
    z={'horizon_day':h,'pair':pair}
    cnt=counts[(counts.horizon_day==h)&(counts.pair==pair)].iloc[0]
    z.update({'source_n':int(cnt.source_n),'source_prevalence':cnt.source_prevalence,'target_n':int(cnt.target_n),'target_prevalence':cnt.target_prevalence})
    for met in ['auroc','average_precision','brier','log_loss','ece','aurc']:
        z[f'{met}_mean']=g[met].mean();z[f'{met}_sd']=g[met].std(ddof=1)
    for method in ['marginal','label_conditional']:
        q=pc[(pc.horizon_day==h)&(pc.analysis_id==pair)&(pc.conformal_method==method)]
        for met in ['coverage','class0_coverage','class1_coverage','avg_set_size','singleton_rate','singleton_accuracy']:
            z[f'{method}_{met}_mean']=q[met].mean();z[f'{method}_{met}_sd']=q[met].std(ddof=1)
    rows.append(z)
pairs=pd.DataFrame(rows);pairs.to_csv(D/'table_oulad_pair_primary.csv',index=False)
# OULAD plot target AUROC per pair
fig,ax=plt.subplots(figsize=(10,6))
for j,h in enumerate([14,56]):
    d=pairs[pairs.horizon_day==h].sort_values('pair');x=np.arange(len(d))+(j-.5)*.22
    ax.errorbar(x,d.auroc_mean,yerr=d.auroc_sd,fmt='o',capsize=3,label=f'Day {h}')
ax.set_xticks(np.arange(len(sorted(pairs.pair.unique()))));ax.set_xticklabels(sorted(pairs.pair.unique()),rotation=35,ha='right')
ax.set_ylabel('Target AUROC');ax.set_xlabel('Matched module–presentation-period pair');ax.set_ylim(.45,.9);ax.grid(axis='y',alpha=.3);ax.legend();fig.tight_layout();fig.savefig(F/'figure_oulad_pair_auroc.png',dpi=300);plt.close(fig)
# OULAD conformal class coverage plot
cm=pd.read_csv(O/'oulad_conformal_macro_by_repeat.csv');cm=cm[(cm.domain=='ood_test')&(cm.aggregation=='equal_pair')]
agg=cm.groupby(['horizon_day','conformal_method'])[['class0_coverage','class1_coverage','avg_set_size']].agg(['mean','std']).reset_index()
fig,ax=plt.subplots(figsize=(9,5.5));labels=[];x=[];y=[];e=[]
for h in [14,56]:
  for method in ['marginal','label_conditional']:
    r=agg[(agg.horizon_day==h)&(agg.conformal_method==method)].iloc[0]
    for cls in [0,1]:
      labels.append(f'D{h}\n{method.replace("label_conditional","label-cond.")}\nclass {cls}')
      x.append(len(x));y.append(r[(f'class{cls}_coverage','mean')]);e.append(r[(f'class{cls}_coverage','std')])
ax.errorbar(x,y,yerr=e,fmt='o',capsize=4);ax.axhline(.9,linestyle='--',linewidth=1);ax.set_xticks(x);ax.set_xticklabels(labels);ax.set_ylim(.65,1.0);ax.set_ylabel('Empirical target coverage');ax.grid(axis='y',alpha=.3);fig.tight_layout();fig.savefig(F/'figure_oulad_conformal.png',dpi=300);plt.close(fig)
# South summaries and distribution figure
sm=pd.read_csv(O/'south_metrics_complete.csv.gz');sc=pd.read_csv(O/'south_conformal_complete.csv.gz')
sm=sm[sm.mapping=='platt'].copy();sm['group']=np.where(sm.scenario.str.startswith('random_three_features_draw_'),'random_three_features',sm.scenario)
sc=sc[(sc.mapping=='platt')&(sc.alpha==.10)].copy();sc['group']=np.where(sc.scenario.str.startswith('random_three_features_draw_'),'random_three_features',sc.scenario)
srows=[]
for group,g in sm.groupby('group'):
  z={'condition':group}
  for met in ['auroc','average_precision','brier','log_loss','ece','hcep_090','aurc','excess_aurc']:
    z[f'{met}_mean']=g[met].mean();z[f'{met}_sd']=g[met].std(ddof=1)
  for method in ['marginal','label_conditional']:
    q=sc[(sc.group==group)&(sc.conformal_method==method)]
    for met in ['coverage','class0_coverage','class1_coverage','avg_set_size','singleton_rate']:
      z[f'{method}_{met}_mean']=q[met].mean();z[f'{method}_{met}_sd']=q[met].std(ddof=1)
  srows.append(z)
pd.DataFrame(srows).to_csv(D/'table_south_primary.csv',index=False)
# plot targeted against random feature-set degradation distribution using per-refit delta from clean
rnd=sm[sm.group=='random_three_features'].copy();tar=sm[sm.group=='targeted_top3'].copy();clean=sm[sm.group=='clean'].set_index('repeat')
metrics_plot=[('auroc','AUROC decrease',-1),('brier','Brier increase',1),('log_loss','Log-loss increase',1),('aurc','AURC increase',1)]
fig,axes=plt.subplots(1,4,figsize=(14,4.5))
for ax,(met,label,sgn) in zip(axes,metrics_plot):
  vals=[]
  for rep,g in rnd.groupby('repeat'):
    vals.extend((sgn*(g[met]-clean.loc[rep,met])).tolist())
  tvals=np.array([sgn*(r[met]-clean.loc[r['repeat'],met]) for _,r in tar.iterrows()])
  ax.boxplot(vals,vert=True,showfliers=False);ax.scatter(np.ones(len(tvals))*1.08,tvals,s=14,alpha=.65,label='Targeted top-3');ax.set_xticks([]);ax.set_title(label);ax.grid(axis='y',alpha=.3)
axes[0].set_ylabel('Degradation relative to clean test data');axes[-1].legend(loc='best');fig.tight_layout();fig.savefig(F/'figure_south_random_targeted.png',dpi=300);plt.close(fig)
# Heart primary and conformal tables
hm=pd.read_csv(O/'heart_metrics_complete.csv.gz');hc=pd.read_csv(O/'heart_conformal_complete.csv.gz');hs=pd.read_csv(O/'heart_selection_complete.csv.gz')
hm=hm[(hm.mapping=='platt')&(~hm.missing_indicators)];hc=hc[(hc.mapping=='platt')&(~hc.missing_indicators)&(hc.alpha==.10)];hs=hs[(hs.mapping=='platt')&(~hs.missing_indicators)&(hs.desired_selection_rate==.80)]
hrows=[]
for (site,domain),g in hm.groupby(['heldout_site','domain']):
  z={'heldout_site':site,'domain':domain}
  for met in ['prevalence','auroc','average_precision','pr_skill','brier','log_loss','ece','hcep_090','aurc']:
    z[f'{met}_mean']=g[met].mean();z[f'{met}_sd']=g[met].std(ddof=1)
  q=hs[(hs.heldout_site==site)&(hs.domain==domain)];z['selection_rate_mean']=q.selection_rate.mean();z['selection_rate_sd']=q.selection_rate.std(ddof=1);z['selective_risk_mean']=q.selective_risk.mean();z['selective_risk_sd']=q.selective_risk.std(ddof=1)
  hrows.append(z)
pd.DataFrame(hrows).to_csv(D/'table_heart_primary.csv',index=False)
hc.groupby(['heldout_site','domain','conformal_method'])[['coverage','class0_coverage','class1_coverage','avg_set_size','singleton_rate','singleton_accuracy','empty_rate']].agg(['mean','std']).to_csv(D/'table_heart_conformal.csv')
# Heart plot OOD conformal
q=hc[hc.domain=='ood_test'].groupby(['heldout_site','conformal_method'])[['class0_coverage','class1_coverage']].mean().reset_index()
fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
for ax,method in zip(axes,['marginal','label_conditional']):
  z=q[q.conformal_method==method];xx=np.arange(len(z));ax.plot(xx,z.class0_coverage,'o-',label='Class 0');ax.plot(xx,z.class1_coverage,'s--',label='Class 1');ax.axhline(.9,linestyle=':',linewidth=1);ax.set_xticks(xx);ax.set_xticklabels(z.heldout_site,rotation=30,ha='right');ax.set_title(method.replace('_',' '));ax.grid(axis='y',alpha=.3)
axes[0].set_ylabel('Target-site empirical coverage');axes[1].legend();fig.tight_layout();fig.savefig(F/'figure_heart_class_conformal.png',dpi=300);plt.close(fig)
# Heart calibration reliability target sites, primary Platt fixed mean bins across refits
rb=pd.read_csv(O/'heart_reliability_bins.csv')
fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharex=True,sharey=True)
for ax,site in zip(axes,['Switzerland','VA Long Beach']):
 z=rb[rb.heldout_site==site].groupby('bin').agg(mean_probability=('mean_probability','mean'),observed_rate=('observed_rate','mean'),n=('n','sum')).reset_index();ax.plot([0,1],[0,1],linestyle='--',linewidth=1);ax.plot(z.mean_probability,z.observed_rate,'o-');ax.set_title(site);ax.set_xlabel('Mean predicted probability');ax.grid(alpha=.3)
axes[0].set_ylabel('Observed prevalence');fig.tight_layout();fig.savefig(F/'figure_heart_reliability.png',dpi=300);plt.close(fig)
print('tables and non-ACS figures built')
