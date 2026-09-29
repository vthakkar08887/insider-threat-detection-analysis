import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
np.random.seed(42)
df=pd.read_csv('/sessions/pensive-amazing-euler/mnt/Downloads/r4.2/chapter4_analysis/user_week.csv')
feats=['login_frequency','off_hours_access','file_transfer_activity','privilege_escalation']
names=['Login frequency','Off-hours access','File-transfer activity','Removable-device connection frequency']
y=df['insider_risk'].astype(int).values; groups=df['user_id'].values
X=df[feats].values.astype(float)
Xz=(X-X.mean(0))/X.std(0)   # standardize only (no log1p)
Xc=sm.add_constant(Xz)
m=sm.Logit(y,Xc).fit(disp=0)
mr=sm.Logit(y,Xc).fit(cov_type='cluster',cov_kwds={'groups':groups},disp=0)
coef=mr.params; se=mr.bse; z=mr.tvalues; p=mr.pvalues
OR=np.exp(coef); lo=np.exp(coef-1.96*se); hi=np.exp(coef+1.96*se)
print('=== RQ1 standardize-ONLY (per-SD) + cluster-robust ===')
labs=['Constant']+names
for i,l in enumerate(labs):
    print(f'{l:40s} B={coef[i]:7.3f} robSE={se[i]:6.3f} z={z[i]:7.2f} p={p[i]:9.2e} OR={OR[i]:6.3f} [{lo[i]:.3f},{hi[i]:.3f}]')
llf=m.llf
print('\ndrop-one LR (df=1):')
for j,f in enumerate(feats):
    keep=[k for k in range(4) if k!=j]
    mrj=sm.Logit(y,sm.add_constant(Xz[:,keep])).fit(disp=0)
    LR=2*(llf-mrj.llf); print(f'  {names[j]:40s} LR={LR:8.2f} p={stats.chi2.sf(LR,1):.2e}')
ll0=sm.Logit(y,np.ones((len(y),1))).fit(disp=0).llf
print(f'\nOmnibus LR chi2(4)={2*(llf-ll0):.2f}')
n=len(y); cox=1-np.exp(2*(ll0-llf)/n); nag=cox/(1-np.exp(2*ll0/n))
print(f'Cox&Snell={cox:.4f} Nagelkerke={nag:.4f} maxCox&Snell={1-np.exp(2*ll0/n):.4f}')
ph=m.predict(Xc)
print(f'Brier={np.mean((ph-y)**2):.5f}')
