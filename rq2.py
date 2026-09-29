import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, recall_score, f1_score
from scipy import stats
np.random.seed(42)
df=pd.read_csv('/sessions/pensive-amazing-euler/mnt/Downloads/r4.2/chapter4_analysis/user_week.csv')
feats=['login_frequency','off_hours_access','file_transfer_activity','privilege_escalation']
y=df['insider_risk'].astype(int).values; groups=df['user_id'].values
X=df[feats].values.astype(float)
gkf=GroupKFold(n_splits=10)
models={'Logistic regression':lambda:LogisticRegression(max_iter=2000),
        'Random forest':lambda:RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1),
        'Gradient boosting':lambda:GradientBoostingClassifier(random_state=42)}
oof={m:np.zeros(len(y)) for m in models}
fold_auc={m:[] for m in models}
folds=list(gkf.split(X,y,groups))
pos_per_fold=[]
for tr,te in folds:
    mu,sd=X[tr].mean(0),X[tr].std(0); sd[sd==0]=1
    Xtr=(X[tr]-mu)/sd; Xte=(X[te]-mu)/sd
    pos_per_fold.append(int(y[te].sum()))
    for name,ctor in models.items():
        clf=ctor().fit(Xtr,y[tr]); p=clf.predict_proba(Xte)[:,1]
        oof[name][te]=p
        try: fold_auc[name].append(roc_auc_score(y[te],p))
        except: fold_auc[name].append(np.nan)
print('positives per fold:',pos_per_fold,'mean=%.1f'%np.mean(pos_per_fold))
# metrics on pooled OOF
def user_boot_ci(y,p,g,n=2000):
    uni=np.unique(g); aucs=[]
    rng=np.random.default_rng(42)
    idxby={u:np.where(g==u)[0] for u in uni}
    for _ in range(n):
        samp=rng.choice(uni,len(uni),replace=True)
        ii=np.concatenate([idxby[u] for u in samp])
        if y[ii].sum()==0 or y[ii].sum()==len(ii): continue
        aucs.append(roc_auc_score(y[ii],p[ii]))
    return np.percentile(aucs,2.5),np.percentile(aucs,97.5)
rows=[]
for name in models:
    p=oof[name]; auc=roc_auc_score(y,p); lo,hi=user_boot_ci(y,p,groups)
    prauc=average_precision_score(y,p); brier=brier_score_loss(y,p)
    rec=recall_score(y,(p>=.5).astype(int)); f1=f1_score(y,(p>=.5).astype(int),zero_division=0)
    rows.append((name,auc,lo,hi,prauc,brier,rec,f1))
    print(f'{name:22s} AUC={auc:.3f} [{lo:.3f},{hi:.3f}] PR-AUC={prauc:.3f} Brier={brier:.5f} Recall={rec:.3f} F1={f1:.3f}')
pd.DataFrame(rows,columns=['model','ROC_AUC','CI_low','CI_high','PR_AUC','Brier','Recall','F1']).to_csv('reanalysis/rq2_metrics.csv',index=False)

# fast DeLong for two correlated ROC
def delong(y,p1,p2):
    y=np.asarray(y); order=np.argsort(-p1-0*p2)
    from scipy.stats import norm
    def midrank(x):
        J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
        while i<N:
            j=i
            while j<N and Z[j]==Z[i]: j+=1
            T[i:j]=0.5*(i+j-1)+1
            i=j
        T2=np.empty(N); T2[J]=T; return T2
    def compute(preds,labels):
        pos=preds[labels==1]; neg=preds[labels==0]; m=len(pos); n=len(neg)
        tx=midrank(pos); ty=midrank(neg); tz=midrank(np.r_[pos,neg])
        auc=(tz[:m].sum()-m*(m+1)/2)/(m*n)
        v01=(tz[:m]-tx)/n; v10=1-(tz[m:]-ty)/m
        return auc,v01,v10,m,n
    a1,v01_1,v10_1,m,n=compute(p1,y); a2,v01_2,v10_2,_,_=compute(p2,y)
    s01=np.cov(np.stack([v01_1,v01_2]))/m; s10=np.cov(np.stack([v10_1,v10_2]))/n
    S=s01+s10
    var=S[0,0]+S[1,1]-2*S[0,1]
    z=(a1-a2)/np.sqrt(var) if var>0 else 0
    p=2*norm.sf(abs(z))
    return a1,a2,z,p
names=list(models)
print('\n=== DeLong (pooled OOF) + Bonferroni alpha=.0167 ===')
del_rows=[]
for i in range(len(names)):
    for j in range(i+1,len(names)):
        a1,a2,z,pp=delong(y,oof[names[i]],oof[names[j]])
        sig='*' if pp<.0167 else 'ns'
        print(f'{names[i]} vs {names[j]}: AUC {a1:.3f} vs {a2:.3f}, z={z:.2f}, p={pp:.2e} [{sig}]')
        del_rows.append((f'{names[i]} vs {names[j]}',a1,a2,z,pp))
pd.DataFrame(del_rows,columns=['comparison','AUC1','AUC2','z','p']).to_csv('reanalysis/rq2_delong.csv',index=False)

# Nadeau-Bengio corrected resampled t on fold AUCs (test train/test ratio ~ 1/9)
print('\n=== Corrected resampled t (Nadeau-Bengio) on fold AUCs ===')
J=10; ratio=1/ (J-1)
for i in range(len(names)):
    for j in range(i+1,len(names)):
        d=np.array(fold_auc[names[i]])-np.array(fold_auc[names[j]])
        m_=d.mean(); v=d.var(ddof=1)
        t=m_/np.sqrt(v*(1/J+ratio)); pval=2*stats.t.sf(abs(t),J-1)
        print(f'{names[i]} vs {names[j]}: mean dAUC={m_:.4f}, t={t:.2f}, p={pval:.3f}')

# LR calibration from OOF
from sklearn.metrics import brier_score_loss
p=np.clip(oof['Logistic regression'],1e-6,1-1e-6); lp=np.log(p/(1-p))
import statsmodels.api as sm
slope=sm.Logit(y,sm.add_constant(lp)).fit(disp=0).params[1]
inter=sm.Logit(y,np.ones((len(y),1)),offset=lp).fit(disp=0).params[0]
print(f'\nLR OOF calibration: intercept={inter:.3f}, slope={slope:.3f}, Brier={brier_score_loss(y,oof["Logistic regression"]):.5f}')
np.save('reanalysis/oof_lr.npy',oof['Logistic regression'])
np.save('reanalysis/oof_gb.npy',oof['Gradient boosting'])
print('saved rq2 outputs')
