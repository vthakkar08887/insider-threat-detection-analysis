import numpy as np, pandas as pd, glob, csv
from datetime import date, datetime
ANS='/sessions/pensive-amazing-euler/mnt/Downloads/answers 2'
# answer-key device Connect ids and file ids
ak_dev=set(); ak_file=set()
for f in glob.glob(ANS+'/r4.2-*/*.csv'):
    for row in csv.reader(open(f)):
        if not row: continue
        typ=row[0]; eid=row[1]
        if typ=='device' and len(row)>=6 and row[5].strip()=='Connect': ak_dev.add(eid)
        if typ=='file': ak_file.add(eid)
print('answer-key device Connect events:',len(ak_dev),'| file events:',len(ak_file))

uw=pd.read_csv('/sessions/pensive-amazing-euler/mnt/Downloads/r4.2/chapter4_analysis/user_week.csv')
# week-start string -> week_index map
wmap=dict(zip(uw['week'],uw['week_index']))
base=date(2009,12,28)
def widx(dt):
    d=datetime.strptime(dt.split()[0],'%m/%d/%Y').date()
    monday=date.fromordinal(d.toordinal()-d.weekday())
    return (monday-base).days//7
insider_weeks=set(map(tuple, uw[uw['insider_risk']==1][['user_id','week_index']].values))
insider_users=set(uw[uw['insider_risk']==1]['user_id'])

# scan device.csv: Connect events for insider users; classify by insider-week + answer-key
tot_conn_iw=0; ak_conn_iw=0
dev_counts_excl={}  # (user,widx)->connect count excluding answer-key events
import csv as _csv
with open('/sessions/pensive-amazing-euler/mnt/Downloads/r4.2/device.csv') as fh:
    r=_csv.reader(fh); next(r)
    for eid,dt,user,pc,act in r:
        if act!='Connect' or user not in insider_users: continue
        wi=widx(dt); key=(user,wi)
        if key in insider_weeks:
            tot_conn_iw+=1
            if eid in ak_dev: ak_conn_iw+=1
            else: dev_counts_excl[key]=dev_counts_excl.get(key,0)+1
print(f'\nDevice Connect events in insider-risk weeks: {tot_conn_iw}')
print(f'  of which answer-key (scripted) events: {ak_conn_iw} ({100*ak_conn_iw/tot_conn_iw:.1f}%)')
print(f'  organic (non-answer-key) events: {tot_conn_iw-ak_conn_iw} ({100*(tot_conn_iw-ak_conn_iw)/tot_conn_iw:.1f}%)')
# how many insider weeks have ZERO organic device connects (i.e., feature entirely scripted)
zero_org=sum(1 for k in insider_weeks if dev_counts_excl.get(k,0)==0)
print(f'  insider-risk weeks with no organic device connections: {zero_org} of {len(insider_weeks)} ({100*zero_org/len(insider_weeks):.1f}%)')
