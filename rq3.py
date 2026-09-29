import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
np.random.seed(42)
df=pd.read_csv('/sessions/pensive-amazing-euler/mnt/Downloads/r4.2/chapter4_analysis/user_week.csv')
feats=['login_frequency','off_hours_access','file_transfer_activity','privilege_escalation']
wk=df['week_index'].values; y=df['insider_risk'].astype(int).values; X=df[feats].values.astype(float); uid=df['user_id'].values
wmin,wmax=wk.min(),wk.max(); span=wmax-wmin+1
tr_end=wmin+int(round(span*0.70)); va_end=wmin+int(round(span*0.85))
train=wk<tr_end; val=(wk>=tr_end)&(wk<va_end); test=wk>=va_end
def part_counts(mask,label):
    n=mask.sum(); pos=y[mask].sum(); ins=len(np.unique(uid[mask & (y==1)]))
    wks=(wk[mask].min(),wk[mask].max())
    print(f'  {label}: weeks {wks[0]}-{wks[1]} | user-weeks={n} | insider-weeks={pos} | distinct insiders={ins}')
print('=== Chronological partitions (70/15/15 by week) ===')
part_counts(train,'Train'); part_counts(val,'Validation'); part_counts(test,'Test')
# standardize by train stats
mu,sd=X[train].mean(0),X[train].std(0); sd[sd==0]=1
Xtr=(X[train]-mu)/sd; Xte=(X[test]-mu)/sd
print('\n=== Ensembles on chronological hold-out (test partition) ===')
res=[]
for name,clf in [('Logistic regression',LogisticRegression(max_iter=2000)),
                 ('Random forest',RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1)),
                 ('Gradient boosting',GradientBoostingClassifier(random_state=42))]:
    clf.fit(Xtr,y[train]); p=clf.predict_proba(Xte)[:,1]
    auc=roc_auc_score(y[test],p); pr=average_precision_score(y[test],p)
    res.append((name,auc,pr)); print(f'  {name:22s} hold-out AUC={auc:.3f}  PR-AUC={pr:.3f}')
pd.DataFrame(res,columns=['model','holdout_AUC','holdout_PRAUC']).to_csv('reanalysis/rq3_holdout.csv',index=False)
print('\nCompare: grouped-CV AUCs were LR .827, RF .883, GB .976 (rq2)')
print('saved rq3_holdout.csv')
