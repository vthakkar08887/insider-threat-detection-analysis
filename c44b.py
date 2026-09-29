import numpy as np, pandas as pd, glob, csv as _csv
from datetime import date, datetime
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
ANS='/sessions/pensive-amazing-euler/mnt/Downloads/answers 2'
ak_dev=set()
for f in glob.glob(ANS+'/r4.2-*/*.csv'):
    for row in _csv.reader(open(f)):
        if row and row[0]=='device' and len(row)>=6 and row[5].strip()=='Connect': ak_dev.add(row[1])
uw=pd.read_csv('/sessions/pensive-amazing-euler/mnt/Downloads/r4.2/chapter4_analysis/user_week.csv')
base=date(2009,12,28)
def widx(dt):
    d=datetime.strptime(dt.split()[0],'%m/%d/%Y').date()
    return (date.fromordinal(d.toordinal()-d.weekday())-base).days//7
insider_users=set(uw[uw['insider_risk']==1]['user_id'])
akcnt={}
with open('/sessions/pensive-amazing-euler/mnt/Downloads/r4.2/device.csv') as fh:
    r=_csv.reader(fh); next(r)
    for eid,dt,user,pc,act in r:
        if act=='Connect' and user in insider_users and eid in ak_dev:
            k=(user,widx(dt)); akcnt[k]=akcnt.get(k,0)+1
# adjusted removable-device feature = original minus answer-key connects (insider weeks only)
adj=uw['privilege_escalation'].values.astype(float).copy()
key=list(zip(uw['user_id'],uw['week_index']))
for i,k in enumerate(key):
    if k in akcnt: adj[i]=max(0,adj[i]-akcnt[k])
feats=['login_frequency','off_hours_access','file_transfer_activity','privilege_escalation']
X=uw[feats].values.astype(float); y=uw['insider_risk'].astype(int).values; g=uw['user_id'].values
Xadj=X.copy(); Xadj[:,3]=adj
gkf=GroupKFold(10); folds=list(gkf.split(X,y,g))
def cv_auc(M,cols):
    oof=np.zeros(len(y))
    for tr,te in folds:
        clf=GradientBoostingClassifier(random_state=42).fit(M[np.ix_(tr,cols)],y[tr])
        oof[te]=clf.predict_proba(M[np.ix_(te,cols)])[:,1]
    return roc_auc_score(y,oof)
print('=== Sensitivity: excluding answer-key (scripted) device events ===')
print(f'Removable-device feature ALONE: original AUC={cv_auc(X,[3]):.3f} -> adjusted AUC={cv_auc(Xadj,[3]):.3f}')
print(f'All four features:              original AUC={cv_auc(X,[0,1,2,3]):.3f} -> adjusted AUC={cv_auc(Xadj,[0,1,2,3]):.3f}')
# mean feature value insider vs non-insider, adjusted
print(f'\nInsider-week mean removable-device: original={X[y==1,3].mean():.2f} -> adjusted={adj[y==1].mean():.2f}')
print(f'Non-insider-week mean: {X[y==0,3].mean():.2f} (unchanged)')
