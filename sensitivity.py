import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
np.random.seed(42)
df=pd.read_csv('/sessions/pensive-amazing-euler/mnt/Downloads/r4.2/chapter4_analysis/user_week.csv')
feats=['login_frequency','off_hours_access','file_transfer_activity','privilege_escalation']
names={'login_frequency':'Login frequency','off_hours_access':'Off-hours access',
       'file_transfer_activity':'File-transfer activity','privilege_escalation':'Removable-device connection frequency'}
y=df['insider_risk'].astype(int).values; g=df['user_id'].values; X=df[feats].values.astype(float)
gkf=GroupKFold(10); folds=list(gkf.split(X,y,g))
def cv_auc(cols):
    oof=np.zeros(len(y))
    for tr,te in folds:
        Xtr=X[np.ix_(tr,cols)]; Xte=X[np.ix_(te,cols)]
        clf=GradientBoostingClassifier(random_state=42).fit(Xtr,y[tr])
        oof[te]=clf.predict_proba(Xte)[:,1]
    return roc_auc_score(y,oof)
print('=== Single-feature grouped-CV AUC (gradient boosting) ===')
for j,f in enumerate(feats):
    print(f'  {names[f]:40s} AUC={cv_auc([j]):.3f}')
print(f'  {"All four features":40s} AUC={cv_auc([0,1,2,3]):.3f}')
print(f'  {"Without removable-device feature":40s} AUC={cv_auc([0,1,2]):.3f}')
print('\n=== Feature-label correspondence (user-week level) ===')
rd=df['privilege_escalation'].values; ft=df['file_transfer_activity'].values
for lbl,mask in [('Insider-risk weeks',y==1),('Non-insider weeks',y==0)]:
    print(f'  {lbl}: removable-device>0 in {100*np.mean(rd[mask]>0):.1f}%, file-copies>0 in {100*np.mean(ft[mask]>0):.1f}%')
