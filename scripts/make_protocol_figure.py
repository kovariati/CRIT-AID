import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=Path(os.environ.get('CRIT_AID_ROOT', Path(__file__).resolve().parents[1])); F=ROOT/'figures'; F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size':11})
fig,ax=plt.subplots(figsize=(12,6.6));ax.set_xlim(0,12);ax.set_ylim(0,6.6);ax.axis('off')
boxes=[
(0.4,4.7,2.4,1.25,'1. Define audit target','Source/target domains\nTarget semantics\nDomain-specific tolerances'),
(3.1,4.7,2.4,1.25,'2. Lock data protocol','Eligibility and estimand\nDeterministic cohorts\nDisjoint partitions'),
(5.8,4.7,2.4,1.25,'3. Fit source pipeline','Primary model\nProbability mapping\nNo target tuning'),
(8.5,4.7,2.8,1.25,'4. Evaluate ID and OOD','Ranking and proper scores\nCalibration\nPaired OOD−ID changes'),
(0.9,2.4,2.8,1.25,'5. Transport uncertainty rules','Selection thresholds\nMarginal and label-conditional\nconformal quantiles'),
(4.5,2.4,2.8,1.25,'6. Stress evidence quality','Matched-budget missingness\nMeasurement corruption\nTarget-definition sensitivity'),
(8.1,2.4,2.8,1.25,'7. Report audit outcome','Selection rate and risk\nSet validity/informativeness\nDomain and subgroup heterogeneity'),
]
for x,y,w,h,title,text in boxes:
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.08',linewidth=1.2,facecolor='white');ax.add_patch(p);ax.text(x+w/2,y+h-.27,title,ha='center',va='center',fontweight='bold');ax.text(x+w/2,y+.43,text,ha='center',va='center',linespacing=1.3)
# arrows top sequence
for a,b in [(0,1),(1,2),(2,3)]:
 x1=boxes[a][0]+boxes[a][2];y1=boxes[a][1]+boxes[a][3]/2;x2=boxes[b][0];y2=boxes[b][1]+boxes[b][3]/2
 ax.add_patch(FancyArrowPatch((x1+.05,y1),(x2-.05,y2),arrowstyle='-|>',mutation_scale=13,linewidth=1.1))
# arrows down / second row
ax.add_patch(FancyArrowPatch((9.9,4.65),(9.6,3.7),arrowstyle='-|>',mutation_scale=13,linewidth=1.1,connectionstyle='arc3,rad=.15'))
ax.add_patch(FancyArrowPatch((8.05,3.0),(7.35,3.0),arrowstyle='-|>',mutation_scale=13,linewidth=1.1))
ax.add_patch(FancyArrowPatch((4.45,3.0),(3.75,3.0),arrowstyle='-|>',mutation_scale=13,linewidth=1.1))
# interpretation band
band=FancyBboxPatch((1.0,.45),10.0,1.05,boxstyle='round,pad=0.05',linewidth=1.2,facecolor='white');ax.add_patch(band)
ax.text(6,1.2,'Interpretation rule',ha='center',fontweight='bold')
ax.text(6,.78,'Transport-stable, transport-sensitive, or indeterminate relative to pre-registered domain-specific tolerances;\nno universal safety threshold is inferred from the benchmark data.',ha='center',va='center')
for x in [2.3,5.9,9.5]:ax.add_patch(FancyArrowPatch((x,2.35),(x,1.55),arrowstyle='-|>',mutation_scale=12,linewidth=1.0))
fig.tight_layout();fig.savefig(F/'figure1_crit_aid_protocol.png',dpi=300,bbox_inches='tight');plt.close(fig)
