import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
np.random.seed(42)

df=pd.read_csv('/sessions/pensive-amazing-euler/mnt/Downloads/r4.2/chapter4_analysis/user_week.csv')
feats=['login_frequency','off_hours_access','file_transfer_activity','privilege_escalation']
names={'login_frequency':'Login frequency','off_hours_access':'Off-hours access',
       'file_transfer_activity':'File-transfer activity','privilege_escalation':'Removable-device connection frequency'}
y=df['insider_risk'].astype(int).values
groups=df['user_id'].values
# log1p + standardize (full-data stats for the descriptive coefficient model)
X=df[feats].values.astype(float)
Xl=np.log1p(X)
Xz=(Xl-Xl.mean(0))/Xl.std(0)
Xc=sm.add_constant(Xz)

# cluster-robust logistic (clustered on user)
m=sm.Logit(y,Xc).fit(disp=0)
mr=sm.Logit(y,Xc).fit(cov_type='cluster',cov_kwds={'groups':groups},disp=0)
coef=mr.params; se=mr.bse; z=mr.tvalues; p=mr.pvalues
OR=np.exp(coef); lo=np.exp(coef-1.96*se); hi=np.exp(coef+1.96*se)
print('=== RQ1: cluster-robust logistic (per-SD, log1p+standardized) ===')
print(f'{"Predictor":40s} {"B":>8s} {"robSE":>7s} {"z":>7s} {"p":>9s} {"OR":>6s} [{"95% CI":>13s}]')
labels=['Constant']+[names[f] for f in feats]
for i,lab in enumerate(labels):
    print(f'{lab:40s} {coef[i]:8.3f} {se[i]:7.3f} {z[i]:7.2f} {p[i]:9.2e} {OR[i]:6.3f} [{lo[i]:.3f}, {hi[i]:.3f}]')

# drop-one likelihood-ratio tests (df=1)
llf_full=m.llf
print('\n=== Drop-one likelihood-ratio tests (df=1) ===')
lr_rows=[]
for j,f in enumerate(feats):
    keep=[k for k in range(len(feats)) if k!=j]
    Xr=sm.add_constant(Xz[:,keep])
    mrj=sm.Logit(y,Xr).fit(disp=0)
    LR=2*(llf_full-mrj.llf); pval=stats.chi2.sf(LR,1)
    lr_rows.append((names[f],LR,pval)); print(f'{names[f]:40s} LR chi2={LR:9.2f}  p={pval:.2e}')

# omnibus
LR_omni=2*(m.llf - sm.Logit(y,sm.add_constant(np.ones((len(y),0)))).fit(disp=0).llf)
print(f'\nOmnibus LR chi2(4) = {LR_omni:.2f}')

# pseudo-R2
print(f'Nagelkerke R2 approx (McFadden={m.prsquared:.4f})')
# Cox-Snell & Nagelkerke
ll0=sm.Logit(y,np.ones((len(y),1))).fit(disp=0).llf
n=len(y)
cox=1-np.exp(2*(ll0-m.llf)/n); nag=cox/(1-np.exp(2*ll0/n))
print(f'Cox & Snell R2 = {cox:.4f} | Nagelkerke R2 = {nag:.4f} | max Cox&Snell = {1-np.exp(2*ll0/n):.4f}')

# Hosmer-Lemeshow (10 groups) on full-data fitted probs
ph=m.predict(Xc)
def hosmer(y,p,g=10):
    order=np.argsort(p); yq=y[order]; pq=p[order]
    idx=np.array_split(np.arange(len(y)),g)
    hl=0
    for ix in idx:
        obs=yq[ix].sum(); exp=pq[ix].sum(); n_=len(ix)
        if exp<=0 or exp>=n_: 
            if exp>0 and exp<n_: hl+=(obs-exp)**2/(exp*(1-exp/n_))
            continue
        hl+=(obs-exp)**2/(exp*(1-exp/n_))
    return hl
HL=hosmer(y,ph.values if hasattr(ph,'values') else ph); pHL=stats.chi2.sf(HL,8)
print(f'\nHosmer-Lemeshow chi2(8) = {HL:.2f}, p = {pHL:.2e}')
print(f'Brier (apparent) = {np.mean((ph-y)**2):.5f}')

pd.DataFrame({'predictor':labels,'B':coef,'robSE':se,'z':z,'p':p,'OR':OR,'CI_low':lo,'CI_high':hi}).to_csv('reanalysis/rq1_coef.csv',index=False)
pd.DataFrame(lr_rows,columns=['predictor','LR_chi2','p']).to_csv('reanalysis/rq1_lr.csv',index=False)
print('\nsaved rq1_coef.csv, rq1_lr.csv')
