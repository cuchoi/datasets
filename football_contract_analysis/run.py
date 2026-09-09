from __future__ import annotations
import json, math, re, unicodedata, zipfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz, process
from scipy import stats
from linearmodels.iv import AbsorbingLS

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw'; OUT=ROOT/'output'
RAW.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
SEASONS=list(range(2017,2022))
LEAGUES={
 'Premier League':'EPL','La Liga':'La liga','Bundesliga':'Bundesliga',
 'Serie A':'Serie A','Ligue 1':'Ligue 1'}
S=requests.Session(); S.headers['User-Agent']='Mozilla/5.0 football-contract-research/1.0'
D={'warnings':[],'sources':{},'counts':{}}

def norm(x):
 x='' if pd.isna(x) else str(x)
 x=unicodedata.normalize('NFKD',x)
 x=''.join(c for c in x if not unicodedata.combining(c)).lower()
 x=x.replace('ø','o').replace('ł','l').replace('đ','d').replace('ß','ss')
 x=re.sub(r'\([^)]*\)$',' ',x); x=re.sub(r'[^a-z0-9]+',' ',x)
 return ' '.join(x.split())

def clubnorm(x):
 x=norm(x)
 repl={'paris saint germain':'psg','manchester utd':'manchester united',
       'bayern munchen':'bayern munich','internazionale':'inter',
       'olympique lyonnais':'lyon','olympique marseille':'marseille'}
 for a,b in repl.items(): x=x.replace(a,b)
 x=re.sub(r'\b(fc|cf|afc|ac|calcio|football club)\b',' ',x)
 return ' '.join(x.split())

def posgroup(x):
 t=set(norm(x).split()); raw=str(x).upper()
 if 'gk' in t or 'goalkeeper' in t: return 'GK'
 if t&{'st','cf','lw','rw'} or 'forward' in t or 'striker' in t or 'F' in raw.split(): return 'FWD'
 if t&{'cam','cm','cdm','lm','rm'} or 'midfield' in t or 'M' in raw.split(): return 'MID'
 if t&{'cb','lb','rb','lwb','rwb'} or 'defender' in t or 'D' in raw.split(): return 'DEF'
 return 'OTHER'

def league_name(x):
 x=norm(x)
 if ('premier league' in x or 'english premier' in x) and 'women' not in x: return 'Premier League'
 if 'spain primera' in x or 'la liga' in x: return 'La Liga'
 if ('bundesliga' in x or 'german 1' in x) and '2 bundesliga' not in x: return 'Bundesliga'
 if 'italian serie a' in x or x=='serie a' or 'serie a tim' in x: return 'Serie A'
 if 'french ligue 1' in x or x=='ligue 1': return 'Ligue 1'
 return None

def get(url,path,minbytes=1000):
 if path.exists() and path.stat().st_size>=minbytes: return path
 r=S.get(url,timeout=180); r.raise_for_status(); path.write_bytes(r.content)
 if path.stat().st_size<minbytes: raise RuntimeError(f'{url}: only {path.stat().st_size} bytes')
 return path

def firstcol(cols,*names):
 low={str(c).lower():c for c in cols}
 for n in names:
  if n.lower() in low:return low[n.lower()]
 return None

def load_contracts():
 frames=[]
 for v in range(18,23):
  p=RAW/f'male_players_{v}.csv'
  url=f'https://raw.githubusercontent.com/eddwebster/football_analytics/master/data/fifa/raw/male_players_{v}.csv'
  get(url,p,100000)
  h=pd.read_csv(p,nrows=0).columns
  idc=firstcol(h,'sofifa_id','player_id'); short=firstcol(h,'short_name'); long=firstcol(h,'long_name')
  club=firstcol(h,'club_name','club'); league=firstcol(h,'league_name'); level=firstcol(h,'league_level')
  exp=firstcol(h,'club_contract_valid_until','club_contract_valid_until_year')
  age=firstcol(h,'age'); pos=firstcol(h,'player_positions'); loan=firstcol(h,'club_loaned_from')
  needed=[c for c in [idc,short,long,club,league,level,exp,age,pos,loan] if c]
  z=pd.read_csv(p,usecols=needed,low_memory=False)
  z['season_start']=v+1999; z['player_id']=pd.to_numeric(z[idc],errors='coerce')
  z['short_name']=z[short].astype(str) if short else ''
  z['long_name']=z[long].astype(str) if long else z['short_name']
  z['club_name']=z[club].fillna('').astype(str); z['league']=z[league].map(league_name)
  z['expiry_year']=pd.to_numeric(z[exp],errors='coerce'); z['age']=pd.to_numeric(z[age],errors='coerce')
  z['position']=z[pos].fillna('').astype(str); z['position_group']=z['position'].map(posgroup)
  z['loaned_from']=z[loan].fillna('').astype(str) if loan else ''
  if level: z=z[pd.to_numeric(z[level],errors='coerce').fillna(1).eq(1)]
  z=z[z.league.notna() & z.player_id.notna() & z.expiry_year.notna() & z.loaned_from.eq('')]
  z['years_remaining']=z.expiry_year-z.season_start
  z=z[z.years_remaining.between(1,6) & z.position_group.ne('GK')]
  z['name_short']=z.short_name.map(norm); z['name_long']=z.long_name.map(norm); z['club_norm']=z.club_name.map(clubnorm)
  z['final_year']=(z.years_remaining==1).astype(int); z['penultimate_year']=(z.years_remaining==2).astype(int)
  z['remaining_group']=np.select([z.years_remaining==1,z.years_remaining==2,z.years_remaining>=3],['1','2','3+'],'other')
  frames.append(z[['player_id','season_start','short_name','long_name','club_name','league','expiry_year','age','position_group','years_remaining','final_year','penultimate_year','remaining_group','name_short','name_long','club_norm']])
 c=pd.concat(frames,ignore_index=True).drop_duplicates(['player_id','season_start'])
 c.player_id=c.player_id.astype(int)
 D['sources']['contracts']={'source':'SoFIFA annual launch snapshots, eddwebster/football_analytics','rows':len(c),'seasons':sorted(c.season_start.unique().tolist())}
 return c

def understat_one(label,code,year):
 url=f'https://understat.com/getLeagueData/{quote(code,safe="")}/{year}'
 r=S.get(url,timeout=120,headers={'X-Requested-With':'XMLHttpRequest','Referer':f'https://understat.com/league/{code.replace(" ","_")}/{year}'})
 r.raise_for_status(); j=r.json(); p=j.get('players',[])
 if isinstance(p,dict): p=list(p.values())
 if not p: raise RuntimeError(f'No players in {url}; keys={list(j)}')
 z=pd.DataFrame(p); z['league']=label; z['season_start']=year; return z

def load_performance():
 frames=[]
 for label,code in LEAGUES.items():
  for y in SEASONS:
   try: frames.append(understat_one(label,code,y))
   except Exception as e:
    D['warnings'].append(f'Understat {label} {y}: {e!r}')
    if label=='Premier League':
     season=f'{y}-{str(y+1)[-2:]}'
     p=RAW/f'understat_{season}.csv'
     try:
      get(f'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/understat/understat_player.csv',p,500)
      z=pd.read_csv(p); z['league']=label; z['season_start']=y; frames.append(z)
     except Exception as e2:D['warnings'].append(f'EPL fallback {season}: {e2!r}')
 if not frames: raise RuntimeError('No performance data')
 p=pd.concat(frames,ignore_index=True,sort=False)
 ren={}
 for a,b in [('player_name','player_name'),('player','player_name'),('time','minutes'),('xG','xg'),('xA','xa'),('xGChain','xg_chain'),('xGBuildup','xg_buildup')]:
  if a in p and b not in p:ren[a]=b
 p=p.rename(columns=ren)
 for x in ['games','minutes','goals','xg','assists','xa','shots','key_passes','xg_chain','xg_buildup']:
  if x not in p:p[x]=np.nan
  p[x]=pd.to_numeric(p[x],errors='coerce')
 p['player_name']=p.player_name.astype(str); p['name_norm']=p.player_name.map(norm)
 p['team_title']=p.get('team_title','').fillna('').astype(str); p['club_norm']=p.team_title.map(clubnorm)
 p['position_us']=p.get('position','').fillna('').map(posgroup)
 p=p[p.minutes.gt(0)].sort_values('minutes',ascending=False).drop_duplicates(['league','season_start','name_norm'])
 D['sources']['performance']={'source':'Understat getLeagueData API; EPL GitHub fallback when needed','rows':len(p),'league_seasons':p.groupby(['league','season_start']).ngroups}
 return p

def match_panel(c,p):
 groups={k:g.reset_index(drop=True) for k,g in c.groupby(['league','season_start'])}
 out=[]; bad=[]
 aliases={'rodri':'rodrigo hernandez cascante','casemiro':'carlos henrique casimiro','fabinho':'fabio henrique tavares','fernandinho':'fernando luiz rosa','jorginho':'jorge luiz frello filho','isco':'francisco roman alarcon suarez'}
 for r in p.itertuples(index=False):
  g=groups.get((r.league,int(r.season_start)))
  if g is None:continue
  targets=[r.name_norm]
  if r.name_norm in aliases:targets.append(aliases[r.name_norm])
  best=None
  for target in targets:
   for col in ['name_long','name_short']:
    hit=process.extractOne(target,g[col].to_dict(),scorer=fuzz.WRatio)
    if hit:
     _,ns,idx=hit; cs=fuzz.token_set_ratio(r.club_norm,g.loc[idx,'club_norm']) if r.club_norm else 0
     score=ns+.08*cs+(3 if r.position_us==g.loc[idx,'position_group'] else 0)
     if best is None or score>best[0]:best=(score,ns,cs,idx)
  if not best or not (best[1]>=90 or (best[1]>=84 and best[2]>=65)):
   bad.append({'player':r.player_name,'league':r.league,'season_start':r.season_start,'scores':best});continue
  q=g.loc[best[3]].to_dict(); q.update(r._asdict()); q['match_name_score']=best[1];q['match_club_score']=best[2];out.append(q)
 m=pd.DataFrame(out).sort_values(['match_name_score','minutes'],ascending=False).drop_duplicates(['player_id','season_start'])
 m=m[m.minutes>=450].copy()
 for x in ['goals','xg','assists','xa','shots','key_passes','xg_chain','xg_buildup']:
  m[x+'_per90']=m[x]*90/m.minutes
 m['expected_contrib_per90']=(m.xg+m.xa)*90/m.minutes
 m['goal_contrib_per90']=(m.goals+m.assists)*90/m.minutes
 m['finishing_over_xg_per90']=(m.goals-m.xg)*90/m.minutes
 m['league_season']=m.league+'__'+m.season_start.astype(str)
 a=m.age-27
 for pg in ['FWD','MID','DEF']:
  ind=(m.position_group==pg).astype(float);m[f'age2_{pg}']=a*a*ind;m[f'age3_{pg}']=a*a*a*ind
 D['counts']['match_rate']=len(m)/len(p);D['counts']['matched_panel']=len(m);D['counts']['players']=m.player_id.nunique();D['counts']['unmatched']=len(bad)
 return m,pd.DataFrame(bad)

def add_sofascore(m):
 try:
  z=RAW/'sofascore.zip';get('https://www.kaggle.com/api/v1/datasets/download/akshankrithick/sofascore-seasonwise-ratings-football-soccer',z,10000)
  d=RAW/'sofascore';d.mkdir(exist_ok=True)
  with zipfile.ZipFile(z) as a:a.extractall(d)
  frames=[]
  for f in d.rglob('*.csv'):
   x=pd.read_csv(f,low_memory=False); cols=x.columns
   nc=firstcol(cols,'player_name','player','name');rc=firstcol(cols,'rating','average_rating','avg_rating','sofascore_rating')
   sc=firstcol(cols,'season','season_name','year');lc=firstcol(cols,'league','league_name','tournament','competition')
   if not nc or not rc:continue
   y=pd.DataFrame({'sofa_name':x[nc].astype(str),'sofascore_rating':pd.to_numeric(x[rc],errors='coerce')})
   if sc:y['season_start']=x[sc].astype(str).str.extract(r'(20\d{2})')[0].astype(float)
   else:
    yy=re.search(r'20\d{2}',f.name);y['season_start']=int(yy.group()) if yy else np.nan
   if lc:y['league']=x[lc].map(league_name)
   else:y['league']=league_name(f.name)
   y['name_norm']=y.sofa_name.map(norm);frames.append(y)
  s=pd.concat(frames,ignore_index=True).dropna(subset=['league','season_start','sofascore_rating'])
  s.season_start=s.season_start.astype(int);s=s.sort_values('sofascore_rating',ascending=False).drop_duplicates(['league','season_start','name_norm'])
  m['name_norm_join']=m.player_name.map(norm);m=m.merge(s[['league','season_start','name_norm','sofascore_rating']],left_on=['league','season_start','name_norm_join'],right_on=['league','season_start','name_norm'],how='left')
  D['sources']['sofascore']={'source':'Kaggle akshankrithick season-wise ratings','rows':len(s),'matched':int(m.sofascore_rating.notna().sum())}
 except Exception as e:
  m['sofascore_rating']=np.nan;D['warnings'].append(f'SofaScore unavailable: {e!r}');D['sources']['sofascore']={'status':'failed','error':repr(e)}
 return m

def detect_events(c):
 rows=[]
 for pid,g in c.sort_values('season_start').groupby('player_id'):
  a=g.to_dict('records')
  for x,y in zip(a,a[1:]):
   if y['season_start']!=x['season_start']+1 or y['club_norm']!=x['club_norm'] or y['expiry_year']<=x['expiry_year']:continue
   rows.append({'event_id':f"{pid}_{x['season_start']}",'player_id':pid,'player':x['short_name'],'league':x['league'],'club':x['club_name'],'position_group':x['position_group'],'pre_season':x['season_start'],'post_season':y['season_start'],'old_expiry':x['expiry_year'],'new_expiry':y['expiry_year'],'old_years_remaining':x['years_remaining'],'remaining_group':x['remaining_group']})
 return pd.DataFrame(rows)

def regress(m):
 rows=[]; agecols=[c for c in m if c.startswith('age2_') or c.startswith('age3_')]
 outcomes=['goals_per90','xg_per90','assists_per90','xa_per90','shots_per90','key_passes_per90','expected_contrib_per90','goal_contrib_per90','finishing_over_xg_per90','minutes','games','sofascore_rating']
 for y in outcomes:
  if y not in m or m[y].notna().sum()<200:continue
  d=m[[y,'final_year','penultimate_year','player_id','league_season','minutes']+agecols].replace([np.inf,-np.inf],np.nan).dropna()
  vc=d.player_id.value_counts();d=d[d.player_id.isin(vc[vc>=2].index)]
  if len(d)<200:continue
  X=d[['final_year','penultimate_year']+agecols].astype(float)
  A=pd.DataFrame({'player':d.player_id.astype(str).astype('category'),'league_season':d.league_season.astype('category')},index=d.index)
  w=None if y in ['minutes','games'] else d.minutes/90
  try:
   r=AbsorbingLS(d[y].astype(float),X,absorb=A,weights=w,drop_absorbed=True).fit(cov_type='clustered',clusters=d.player_id,debiased=True)
   for term in ['final_year','penultimate_year']:
    b=float(r.params[term]);se=float(r.std_errors[term]);rows.append({'outcome':y,'term':term,'coefficient':b,'std_error':se,'ci_low':b-1.96*se,'ci_high':b+1.96*se,'p_value':float(r.pvalues[term]),'n':int(r.nobs),'players':d.player_id.nunique(),'mean':d[y].mean(),'pct_mean':100*b/d[y].mean() if d[y].mean()!=0 else np.nan})
  except Exception as e:D['warnings'].append(f'Regression {y}: {e!r}')
 return pd.DataFrame(rows)

def event_outputs(m,e):
 if e.empty:return pd.DataFrame(),pd.DataFrame()
 ix=m.set_index(['player_id','season_start']);out=[]
 outcomes=['goals_per90','xg_per90','assists_per90','xa_per90','expected_contrib_per90','minutes','games','sofascore_rating']
 for r in e.to_dict('records'):
  a=(r['player_id'],r['pre_season']);b=(r['player_id'],r['post_season'])
  if a not in ix.index or b not in ix.index:continue
  x=ix.loc[a];y=ix.loc[b]
  if isinstance(x,pd.DataFrame):x=x.iloc[0]
  if isinstance(y,pd.DataFrame):y=y.iloc[0]
  z=dict(r);z['performance_player']=x.player_name
  for o in outcomes:z['pre_'+o]=x.get(o);z['post_'+o]=y.get(o);z['change_'+o]=y.get(o)-x.get(o) if pd.notna(x.get(o)) and pd.notna(y.get(o)) else np.nan
  out.append(z)
 q=pd.DataFrame(out);summ=[]
 if not q.empty:
  for group,g in [('all',q)]+[(str(k),v) for k,v in q.groupby('remaining_group')]:
   for o in outcomes:
    s=pd.to_numeric(g['change_'+o],errors='coerce').dropna();n=len(s)
    if not n:continue
    se=s.std(ddof=1)/math.sqrt(n) if n>1 else np.nan;p=stats.ttest_1samp(s,0).pvalue if n>1 else np.nan
    summ.append({'remaining_group':group,'outcome':o,'n':n,'mean_pre':g['pre_'+o].mean(),'mean_post':g['post_'+o].mean(),'mean_change':s.mean(),'std_error':se,'p_value':p})
 return q,pd.DataFrame(summ)

def injuries(m,e):
 try:
  base='https://media.githubusercontent.com/media/salimt/football-datasets/main/datalake/transfermarkt'
  ip=get(base+'/player_injuries/player_injuries.csv',RAW/'player_injuries.csv',100000)
  pp=get(base+'/player_profiles/player_profiles.csv',RAW/'player_profiles.csv',100000)
  inj=pd.read_csv(ip,low_memory=False);prof=pd.read_csv(pp,low_memory=False)
  idp=firstcol(prof.columns,'player_id');np_=firstcol(prof.columns,'player_name');idi=firstcol(inj.columns,'player_id');sc=firstcol(inj.columns,'season');dc=firstcol(inj.columns,'days_missed');gc=firstcol(inj.columns,'games_missed');rc=firstcol(inj.columns,'injury_reason')
  names=prof[[idp,np_]].rename(columns={idp:'tm_id',np_:'tm_name'});names['tm_id']=pd.to_numeric(names.tm_id,errors='coerce');names['name_norm_tm']=names.tm_name.map(norm)
  inj['tm_id']=pd.to_numeric(inj[idi],errors='coerce');inj['season_start']=inj[sc].astype(str).str.extract(r'(\d{2})/')[0].astype(float)+2000
  inj['days']=pd.to_numeric(inj[dc].astype(str).str.extract(r'(\d+)')[0],errors='coerce') if dc else np.nan
  inj['missed']=pd.to_numeric(inj[gc].astype(str).str.extract(r'(\d+)')[0],errors='coerce') if gc else np.nan
  reason=inj[rc].fillna('').map(norm) if rc else pd.Series('',index=inj.index);inj=inj[~reason.str.contains(r'\brest\b|suspension|personal reason',regex=True)]
  inj=inj.merge(names,on='tm_id',how='left');a=inj.groupby(['name_norm_tm','season_start'],as_index=False).agg(injury_events=('tm_id','size'),injury_days=('days','sum'),injury_games_missed=('missed','sum'))
  m['name_norm_injury']=m.player_name.map(norm);m=m.merge(a,left_on=['name_norm_injury','season_start'],right_on=['name_norm_tm','season_start'],how='left')
  universe=set(names.name_norm_tm.dropna());covered=m.name_norm_injury.isin(universe)
  for col in ['injury_events','injury_days','injury_games_missed']:m.loc[covered,col]=m.loc[covered,col].fillna(0)
  D['sources']['injuries']={'source':'salimt/football-datasets Transfermarkt histories','panel_name_coverage':float(covered.mean()),'nonzero_player_seasons':int(m.injury_events.fillna(0).gt(0).sum())}
  ix=m.set_index(['player_id','season_start']);rows=[]
  for r in e.to_dict('records'):
   pre=(r['player_id'],r['pre_season']);post=(r['player_id'],r['post_season'])
   if pre not in ix.index or post not in ix.index:continue
   x=ix.loc[pre];y=ix.loc[post]
   if isinstance(x,pd.DataFrame):x=x.iloc[0]
   if isinstance(y,pd.DataFrame):y=y.iloc[0]
   if pd.isna(x.get('injury_events')) or pd.isna(y.get('injury_events')):continue
   z=dict(r);z['performance_player']=x.player_name
   for o in ['injury_events','injury_days','injury_games_missed']:z['pre_'+o]=x[o];z['post_'+o]=y[o];z['change_'+o]=y[o]-x[o]
   rows.append(z)
  q=pd.DataFrame(rows);summ=[]
  if not q.empty:
   for group,g in [('all',q)]+[(str(k),v) for k,v in q.groupby('remaining_group')]:
    for o in ['injury_events','injury_days','injury_games_missed']:
     s=g['change_'+o].dropna();n=len(s);summ.append({'remaining_group':group,'outcome':o,'n':n,'mean_pre':g['pre_'+o].mean(),'mean_post':g['post_'+o].mean(),'mean_change':s.mean(),'p_value':stats.ttest_1samp(s,0).pvalue if n>1 else np.nan})
  return m,q,pd.DataFrame(summ)
 except Exception as ex:
  D['warnings'].append(f'Injuries unavailable: {ex!r}');D['sources']['injuries']={'status':'failed','error':repr(ex)};return m,pd.DataFrame(),pd.DataFrame()

def report(m,e,r,es,ins):
 def table(x):return 'No estimates generated.' if x.empty else x.round(4).to_markdown(index=False)
 lines=['# Football contract-year analysis: executed results','',
 f"Panel: **{len(m):,} player-seasons**, **{m.player_id.nunique():,} players**, {m.groupby(['league','season_start']).ngroups} league-seasons; at least 450 minutes.",
 f"Contract observations: **{int(m.final_year.sum()):,} final-year**, **{int(m.penultimate_year.sum()):,} penultimate-year**. Detected same-club expiry increases: **{len(e):,}**.",'',
 '## Fixed-effects regression','Player fixed effects, league-season fixed effects, position-specific nonlinear age controls; clustered standard errors by player. Per-90 outcomes weighted by minutes. Baseline is 3+ years remaining.','',table(r),'',
 '## Descriptive pre/post around detected extension',table(es),'','## Injury pre/post',table(ins),'',
 '## Interpretation','This is not a true regression discontinuity: renewal timing is chosen by clubs and players. The fixed-effects association asks whether the same player performs differently with one or two years left, conditional on league-season shocks and an ageing curve. The pre/post table remains vulnerable to clubs renewing players after unusually strong seasons and to regression to the mean.','',
 'SofaScore and injury sections are included only when their public datasets downloaded and matched successfully. Comparable historical kilometres run and high-speed running are not publicly available across the five leagues, so those metrics are not invented.','',
 '## Diagnostics','```json',json.dumps(D,indent=2,default=str),'```']
 return '\n'.join(lines)

def main():
 c=load_contracts();p=load_performance();m,bad=match_panel(c,p);m=add_sofascore(m);e=detect_events(c);m,ie,ins=injuries(m,e);r=regress(m);ep,es=event_outputs(m,e)
 D['counts'].update({'contracts':len(c),'performance':len(p),'panel':len(m),'final_year':int(m.final_year.sum()),'penultimate_year':int(m.penultimate_year.sum()),'events':len(e),'events_with_performance':len(ep),'regression_rows':len(r)})
 c.to_csv(OUT/'contract_snapshots.csv',index=False);p.to_csv(OUT/'performance.csv',index=False);m.to_csv(OUT/'analysis_panel.csv',index=False);bad.to_csv(OUT/'unmatched.csv',index=False);e.to_csv(OUT/'extension_events.csv',index=False);r.to_csv(OUT/'regression_results.csv',index=False);ep.to_csv(OUT/'extension_pre_post.csv',index=False);es.to_csv(OUT/'extension_pre_post_summary.csv',index=False);ie.to_csv(OUT/'injury_pre_post.csv',index=False);ins.to_csv(OUT/'injury_summary.csv',index=False)
 (OUT/'diagnostics.json').write_text(json.dumps(D,indent=2,default=str));(OUT/'report.md').write_text(report(m,e,r,es,ins))
 print(json.dumps(D['counts'],indent=2));print(r.to_string(index=False));print(es.to_string(index=False));print(ins.to_string(index=False))
if __name__=='__main__':main()
