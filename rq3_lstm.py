import numpy as np, pandas as pd, random, os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from sklearn.metrics import roc_auc_score
SEED=42; np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)
WINDOW=4; UNITS=32; DROPOUT=0.2; EPOCHS=30; PATIENCE=5; BATCH=256; LR=0.001
df=pd.read_csv('/sessions/pensive-amazing-euler/mnt/Downloads/r4.2/chapter4_analysis/user_week.csv').sort_values(['user_id','week_index'])
feats=['login_frequency','off_hours_access','file_transfer_activity','privilege_escalation']
uid=df['user_id'].values; wk=df['week_index'].values; y=df['insider_risk'].astype(int).values
wmin,wmax=wk.min(),wk.max(); span=wmax-wmin+1
tr_end=wmin+int(round(span*0.70)); va_end=wmin+int(round(span*0.85))
train=wk<tr_end; val=(wk>=tr_end)&(wk<va_end); test=wk>=va_end
mu=df.loc[train,feats].mean().values; sd=df.loc[train,feats].std().replace(0,1).values
Xall=((df[feats].values-mu)/sd).astype('float32')
pos={(uid[i],wk[i]):i for i in range(len(df))}
def make(mask):
    Xs=[];ys=[]
    for i in np.where(mask)[0]:
        seq=[Xall[pos[(uid[i],w)]] if (uid[i],w) in pos else np.zeros(len(feats),'float32') for w in range(wk[i]-WINDOW+1,wk[i]+1)]
        Xs.append(seq); ys.append(y[i])
    return np.array(Xs,'float32'), np.array(ys)
Xtr,ytr=make(train); Xva,yva=make(val); Xte,yte=make(test)
print('shapes tr/va/te:',Xtr.shape,Xva.shape,Xte.shape,'test positives',int(yte.sum()))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
m=Sequential([LSTM(UNITS,input_shape=(WINDOW,len(feats))),Dropout(DROPOUT),Dense(1,activation='sigmoid')])
m.compile(optimizer=tf.keras.optimizers.Adam(LR),loss='binary_crossentropy')
cw={0:1.0,1:float((ytr==0).sum()/max(1,(ytr==1).sum()))}
es=EarlyStopping(monitor='val_loss',patience=PATIENCE,restore_best_weights=True)
h=m.fit(Xtr,ytr,validation_data=(Xva,yva),epochs=EPOCHS,batch_size=BATCH,class_weight=cw,callbacks=[es],verbose=0)
print('epochs trained:',len(h.history['loss']))
p=m.predict(Xte,verbose=0).ravel(); auc=roc_auc_score(yte,p)
rng=np.random.default_rng(SEED); b=[]
for _ in range(2000):
    ii=rng.integers(0,len(yte),len(yte))
    if 0<yte[ii].sum()<len(ii): b.append(roc_auc_score(yte[ii],p[ii]))
lo,hi=np.percentile(b,2.5),np.percentile(b,97.5)
print(f'LSTM hold-out ROC-AUC = {auc:.3f}  95% bootstrap CI [{lo:.3f}, {hi:.3f}]')
print(f'HYPERPARAMS: window={WINDOW}wk, 1 LSTM layer x {UNITS} units, dropout={DROPOUT}, epochs<= {EPOCHS} (early stop patience {PATIENCE}), batch={BATCH}, optimizer=Adam, lr={LR}, seed={SEED}')
import json; json.dump({'auc':round(float(auc),3),'lo':round(float(lo),3),'hi':round(float(hi),3),'epochs':len(h.history['loss'])},open('reanalysis/rq3_lstm.json','w'))
