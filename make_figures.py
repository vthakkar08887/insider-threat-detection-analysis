#!/usr/bin/env python3
"""Generate three APA-style grayscale artifact diagrams (Figures 1-3) as PNGs.
Rev5: privilege-escalation feature relabeled to removable-device connection frequency;
Fig 2 class-imbalance stage moved above modeling layer; Fig 3 ensembles added to the
chronological hold-out path and decision label corrected."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams['font.family']='serif'
plt.rcParams['font.serif']=['DejaVu Serif']

INK='#1a1a1a'; FILL='#f2f2f2'; FILL2='#e0e0e0'; WHITE='#ffffff'

def box(ax,x,y,w,h,text,fc=FILL,fs=11,bold=False):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.06',
                     linewidth=1.3,edgecolor=INK,facecolor=fc)
    ax.add_patch(p)
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=INK,
            wrap=True,fontweight='bold' if bold else 'normal')
def arrow(ax,x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=14,
                 linewidth=1.3,color=INK,shrinkA=1,shrinkB=1))

# ---------- Figure 1: Conceptual framework ----------
fig,ax=plt.subplots(figsize=(7.6,3.6)); ax.set_xlim(0,12.6); ax.set_ylim(0,6.2); ax.axis('off')
box(ax,0.15,1.7,4.5,3.3,'',fc=WHITE)
ax.text(2.4,4.65,'Behavioral Indicators',ha='center',fontsize=10.5,fontweight='bold',color=INK)
for k,lbl in enumerate(['Login frequency','Off-hours access','File-transfer activity','Removable-device connection frequency']):
    ax.text(2.4,4.0-k*0.58,u'• '+lbl,ha='center',fontsize=7.5,color=INK)
box(ax,5.35,2.5,2.9,1.8,'Predictive\nModeling\n(LR, RF, GB, LSTM)',fc=FILL,fs=10,bold=True)
box(ax,9.2,2.5,2.9,1.8,'Insider-Risk\nClassification\n(user-week)',fc=FILL2,fs=10,bold=True)
arrow(ax,4.65,3.4,5.35,3.4); arrow(ax,8.25,3.4,9.2,3.4)
ax.text(6.3,1.05,'Class-imbalance handling (SMOTE / ADASYN)',ha='center',fontsize=8.5,style='italic',color=INK)
plt.tight_layout(); plt.savefig('fig1_conceptual.png',dpi=200,bbox_inches='tight',facecolor='white'); plt.close()

# ---------- Figure 2: Artifact architecture / end-to-end workflow ----------
fig,ax=plt.subplots(figsize=(7.2,5.6)); ax.set_xlim(0,10); ax.set_ylim(0,15); ax.axis('off')
stages=[
 ('Data Sources\nCERT r4.2: logon, file, device logs + answer key',FILL2),
 ('Ingestion & Aggregation\nrecords aggregated to the user-week level',FILL),
 ('Feature Engineering\nlogin frequency, off-hours access,\nfile-transfer activity, removable-device connection frequency',FILL),
 ('Labeling\nbinary insider-risk label from official answer key',FILL),
 ('Class-Imbalance Handling\nSMOTE / ADASYN applied within training folds before model fitting',FILL),
 ('Modeling Layer\nlogistic regression, random forest,\ngradient boosting, LSTM',FILL),
 ('Evaluation Layer\nROC-AUC, DeLong test, bootstrap CIs, recall, F1',FILL),
 ('Outputs\nfeature table, trained models, performance tables, ROC results',FILL2),
]
y=13.5; h=1.25; w=8.4; x=0.8
centers=[]
for txt,fc in stages:
    box(ax,x,y,w,h,txt,fc=fc,fs=8.8,bold=txt.split('\n')[0] in ('Data Sources','Outputs'))
    centers.append(y); y-=1.62
for i in range(len(centers)-1):
    arrow(ax,x+w/2,centers[i],x+w/2,centers[i+1]+h)
plt.tight_layout(); plt.savefig('fig2_architecture.png',dpi=200,bbox_inches='tight',facecolor='white'); plt.close()

# ---------- Figure 3: Modeling and evaluation pipeline ----------
fig,ax=plt.subplots(figsize=(7.2,4.6)); ax.set_xlim(0,14); ax.set_ylim(0,9); ax.axis('off')
box(ax,0.3,3.6,2.5,1.8,'User-week\nfeature table',fc=FILL2,fs=9.5,bold=True)
# two evaluation protocols
box(ax,3.5,6.2,3.6,1.6,'Ensembles + LR\n10-fold cross-validation',fc=FILL,fs=9.0)
box(ax,3.5,1.2,3.6,1.6,'LSTM + ensembles\nchronological 70/15/15 hold-out',fc=FILL,fs=9.0)
arrow(ax,2.8,4.9,3.5,7.0); arrow(ax,2.8,4.1,3.5,2.0)
# resampling note
ax.text(5.3,4.5,'SMOTE / ADASYN\n(training folds only)',ha='center',fontsize=8.3,style='italic',color=INK)
# inference methods
box(ax,7.9,6.2,3.0,1.6,"DeLong's test\n(correlated ROC)",fc=FILL,fs=9.0)
box(ax,7.9,1.2,3.0,1.6,'Bootstrap 95% CIs\n(held-out test set)',fc=FILL,fs=9.0)
arrow(ax,7.1,7.0,7.9,7.0); arrow(ax,7.1,2.0,7.9,2.0)
box(ax,11.1,3.2,2.8,2.6,'ROC-AUC, recall, F1;\nhypothesis decisions\n(RQ1–RQ3);\ndescriptive comparison\n(RQ4)',fc=FILL2,fs=8.5,bold=True)
arrow(ax,10.9,7.0,12.5,5.8); arrow(ax,10.9,2.0,12.5,3.2)
plt.tight_layout(); plt.savefig('fig3_pipeline.png',dpi=200,bbox_inches='tight',facecolor='white'); plt.close()
print('figures written: fig1_conceptual.png, fig2_architecture.png, fig3_pipeline.png')
