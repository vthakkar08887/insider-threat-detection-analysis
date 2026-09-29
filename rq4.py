import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score,f1_score,precision_score,average_precision_score,brier_score_loss
from imblearn.over_sampling import SMOTE, ADASYN
np.random.seed(42)
df=pd.read_csv('/sessions/pensive-amazing-euler/mnt/Downloads/r4.2/chapter4_analysis/user_week.csv')
feats=['login_frequency','off_hours_access','file_transfer_activity','privilege_escalation']
y=df['insider_risk'].astype(int).values; groups=df['user_id'].values; X=df[feats].values.astype(float)
gkf=GroupKFold(n_splits=10); folds=list(gkf.split(X,y,groups))
def run(sampler=None, thresh_shift=False):
    oof=np.zeros(len(y)); pred=np.zeros(len(y),int)
    for tr,te in folds:
        mu,sd=X[tr].mean(0),X[tr].std(0); sd[sd==0]=1
        Xtr=(X[tr]-mu)/sd; Xte=(X[te]-mu)/sd; ytr=y[tr]
        if sampler is not None:
            Xtr,ytr=sampler(random_state=42).fit_resample(Xtr,ytr)
        clf=GradientBoostingClassifier(random_state=42).fit(Xtr,ytr)
        pte=clf.predict_proba(Xte)[:,1]; oof[te]=pte
        if thresh_shift:
            ptr=clf.predict_proba((X[tr]-mu)/sd)[:,1]
            ths=np.unique(ptr); bestf1,bestt=0,0.5
            for t in ths[::max(1,len(ths)//200)]:
                f=f1_score(y[tr],(ptr>=t).astype(int),zero_division=0)
                if f>bestf1: bestf1,bestt=f,t
            pred[te]=(pte>=bestt).astype(int)
        else:
            pred[te]=(pte>=0.5).astype(int)
    return oof,pred
rows=[]
for label,samp,ts in [('No resampling (baseline)',None,False),('SMOTE',SMOTE,False),
                      ('ADASYN',ADASYN,False),('No resampling + F1-max threshold',None,True)]:
    oof,pred=run(samp,ts)
    rec=recall_score(y,pred); f1=f1_score(y,pred,zero_division=0); prec=precision_score(y,pred,zero_division=0)
    prauc=average_precision_score(y,oof); brier=brier_score_loss(y,oof)
    rows.append((label,rec,f1,prec,prauc,brier))
    print(f'{label:34s} Recall={rec:.3f} F1={f1:.3f} Precision={prec:.3f} PR-AUC={prauc:.3f} Brier={brier:.5f}')
pd.DataFrame(rows,columns=['Condition','Recall','F1','Precision','PR_AUC','Brier']).to_csv('reanalysis/rq4_resample.csv',index=False)
print('saved rq4_resample.csv')
