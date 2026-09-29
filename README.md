# Insider-Threat Detection — Analysis Code (CERT r4.2)

Reproducible analysis for the dissertation "Predictive Modeling Approaches for
Insider-Threat Detection Using Behavioral Indicators Derived from Enterprise
Access-Log Data." All scripts use random_state = 42.

## Inputs
- user_week.csv  (aggregated user-week feature table with binary insider_risk label)
- CERT r4.2 raw logs (device.csv, file.csv, logon.csv) and the answers/ folder

## Scripts
- rq1_nolog.py    RQ1: per-SD cluster-robust logistic regression, drop-one LR tests
- rq2.py          RQ2: user-grouped 10-fold CV; ROC-AUC/PR-AUC/Brier; DeLong; corrected resampled t
- rq3.py          RQ3: chronological 70/15/15 hold-out; ensembles; partition counts
- rq4.py          RQ4: resampling (SMOTE/ADASYN) vs F1-max threshold shift
- sensitivity.py  single-feature discrimination + feature-label correspondence
- c44.py, c44b.py answer-key overlap and exclusion sensitivity
- make_figures.py Figures 1-3

## Environment
Python 3; numpy, pandas, scikit-learn, statsmodels, imbalanced-learn, matplotlib.
(The LSTM for RQ3 requires TensorFlow/PyTorch and is run separately.)
