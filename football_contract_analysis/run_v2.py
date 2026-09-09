"""Second-pass runner: fixes wrapper, minutes regression, SofaScore parsing and injury URLs."""
import math, re, zipfile
import numpy as np
import pandas as pd
from scipy import stats
from linearmodels.iv import AbsorbingLS
import run as b


def parse_season(value):
    x=b.norm(value)
    m=re.search(r'(20\d{2})',x)
    if m: return int(m.group(1))
    m=re.search(r'(?<!\d)(1[7-9]|2[0-4])\D+(1[8-9]|2[0-5])(?!\d)',x)
    return 2000+int(m.group(1)) if m else np.nan


def league_path(value):
    x=b.norm(value)
    if 'premier' in x or re.search(r'\bepl\b|\beng\b',x): return 'Premier League'
    if 'la liga' in x or 'laliga' in x or re.search(r'\bspa\b',x): return 'La Liga'
    if 'bundesliga' in x or re.search(r'\bger\b',x): return 'Bundesliga'
    if 'serie a' in x or 'seriea' in x or re.search(r'\bita\b',x): return 'Serie A'
    if 'ligue 1' in x or 'ligue1' in x or re.search(r'\bfra\b',x): return 'Ligue 1'
    return b.league_name(value)


def contains_col(cols,*parts):
    for col in cols:
        name=b.norm(col)
        if any(part in name for part in parts): return col
    return None


def add_sofascore(panel):
    try:
        archive=b.RAW/'sofascore.zip'
        b.get('https://www.kaggle.com/api/v1/datasets/download/akshankrithick/sofascore-seasonwise-ratings-football-soccer',archive,10000)
        directory=b.RAW/'sofascore'; directory.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z: z.extractall(directory)
        frames=[]; schemas=[]
        files=[f for f in directory.rglob('*') if f.suffix.lower()=='.csv']
        for f in files:
            try: data=pd.read_csv(f,sep=None,engine='python')
            except Exception: data=pd.read_csv(f)
            cols=data.columns.tolist()
            schemas.append({'file':str(f.relative_to(directory)),'columns':cols,'rows':len(data)})
            name_col=b.firstcol(cols,'player_name','player','name') or contains_col(cols,'player name','player')
            rating_col=b.firstcol(cols,'rating','average_rating','avg_rating','sofascore_rating') or contains_col(cols,'rating')
            season_col=b.firstcol(cols,'season','season_name','year') or contains_col(cols,'season')
            league_col=b.firstcol(cols,'league','league_name','tournament','competition') or contains_col(cols,'league','tournament','competition')
            if name_col is None or rating_col is None: continue
            out=pd.DataFrame({'sofa_name':data[name_col].astype(str),'sofascore_rating':pd.to_numeric(data[rating_col],errors='coerce')})
            out['season_start']=data[season_col].map(parse_season) if season_col else parse_season(str(f.relative_to(directory)))
            out['league']=data[league_col].map(b.league_name) if league_col else league_path(str(f.relative_to(directory)))
            out['name_norm']=out.sofa_name.map(b.norm); frames.append(out)
        b.D['sources']['sofascore_schema']={'files':len(files),'examples':schemas[:15]}
        if not frames: raise ValueError('No recognizable player/rating columns')
        sofa=pd.concat(frames,ignore_index=True).dropna(subset=['league','season_start','sofascore_rating'])
        sofa.season_start=sofa.season_start.astype(int)
        sofa=sofa[sofa.season_start.isin(b.SEASONS)].sort_values('sofascore_rating',ascending=False).drop_duplicates(['league','season_start','name_norm'])
        panel['name_norm_sofa']=panel.player_name.map(b.norm)
        panel=panel.merge(sofa[['league','season_start','name_norm','sofascore_rating']],left_on=['league','season_start','name_norm_sofa'],right_on=['league','season_start','name_norm'],how='left')
        b.D['sources']['sofascore']={'source':'Kaggle akshankrithick','rows':len(sofa),'matched':int(panel.sofascore_rating.notna().sum()),'match_rate':float(panel.sofascore_rating.notna().mean())}
    except Exception as exc:
        panel['sofascore_rating']=np.nan
        b.D['warnings'].append(f'SofaScore unavailable: {exc!r}')
        b.D['sources']['sofascore']={'status':'failed','error':repr(exc)}
    return panel


def regress(panel):
    rows=[]
    agecols=[c for c in panel if c.startswith('age2_') or c.startswith('age3_')]
    outcomes=['goals_per90','xg_per90','assists_per90','xa_per90','shots_per90','key_passes_per90','expected_contrib_per90','goal_contrib_per90','finishing_over_xg_per90','minutes','games','sofascore_rating']
    for outcome in outcomes:
        if outcome not in panel or panel[outcome].notna().sum()<200: continue
        cols=[outcome,'final_year','penultimate_year','player_id','league_season']+([] if outcome=='minutes' else ['minutes'])+agecols
        data=panel[cols].replace([np.inf,-np.inf],np.nan).dropna()
        counts=data.player_id.value_counts(); data=data[data.player_id.isin(counts[counts>=2].index)]
        if len(data)<200: continue
        X=data[['final_year','penultimate_year']+agecols].astype(float)
        absorb=pd.DataFrame({'player':data.player_id.astype(str).astype('category'),'league_season':data.league_season.astype('category')},index=data.index)
        weights=None if outcome in ['minutes','games'] else data.minutes/90
        try:
            result=AbsorbingLS(data[outcome].astype(float),X,absorb=absorb,weights=weights,drop_absorbed=True).fit(cov_type='clustered',clusters=data.player_id,debiased=True)
            for term in ['final_year','penultimate_year']:
                coef=float(result.params[term]); se=float(result.std_errors[term])
                rows.append({'outcome':outcome,'term':term,'coefficient':coef,'std_error':se,'ci_low':coef-1.96*se,'ci_high':coef+1.96*se,'p_value':float(result.pvalues[term]),'n':int(result.nobs),'players':data.player_id.nunique(),'mean':data[outcome].mean(),'pct_mean':100*coef/data[outcome].mean() if data[outcome].mean()!=0 else np.nan})
        except Exception as exc: b.D['warnings'].append(f'Regression {outcome}: {exc!r}')
    return pd.DataFrame(rows)


def injuries(panel,events):
    try:
        base='https://raw.githubusercontent.com/salimt/football-datasets/main/datalake/transfermarkt'
        injury_path=b.get(base+'/player_injuries/player_injuries.csv',b.RAW/'player_injuries.csv',100000)
        profile_path=b.get(base+'/player_profiles/player_profiles.csv',b.RAW/'player_profiles.csv',100000)
        inj=pd.read_csv(injury_path,low_memory=False); prof=pd.read_csv(profile_path,low_memory=False)
        idp=b.firstcol(prof.columns,'player_id'); namep=b.firstcol(prof.columns,'player_name')
        idi=b.firstcol(inj.columns,'player_id'); season=b.firstcol(inj.columns,'season')
        days=b.firstcol(inj.columns,'days_missed'); games=b.firstcol(inj.columns,'games_missed'); reason=b.firstcol(inj.columns,'injury_reason')
        names=prof[[idp,namep]].rename(columns={idp:'tm_id',namep:'tm_name'})
        names.tm_id=pd.to_numeric(names.tm_id,errors='coerce'); names['name_norm_tm']=names.tm_name.map(b.norm)
        inj['tm_id']=pd.to_numeric(inj[idi],errors='coerce')
        inj['season_start']=pd.to_numeric(inj[season].astype(str).str.extract(r'(\d{2})/')[0],errors='coerce')+2000
        inj['days']=pd.to_numeric(inj[days].astype(str).str.extract(r'(\d+)')[0],errors='coerce')
        inj['missed']=pd.to_numeric(inj[games].astype(str).str.extract(r'(\d+)')[0],errors='coerce')
        reasons=inj[reason].fillna('').map(b.norm)
        inj=inj[~reasons.str.contains(r'\brest\b|suspension|personal reason',regex=True)]
        inj=inj.merge(names,on='tm_id',how='left')
        agg=inj.groupby(['name_norm_tm','season_start'],as_index=False).agg(injury_events=('tm_id','size'),injury_days=('days','sum'),injury_games_missed=('missed','sum'))
        panel['name_norm_injury']=panel.player_name.map(b.norm)
        panel=panel.merge(agg,left_on=['name_norm_injury','season_start'],right_on=['name_norm_tm','season_start'],how='left')
        universe=set(names.name_norm_tm.dropna()); covered=panel.name_norm_injury.isin(universe)
        for col in ['injury_events','injury_days','injury_games_missed']: panel.loc[covered,col]=panel.loc[covered,col].fillna(0)
        b.D['sources']['injuries']={'source':'salimt/football-datasets','panel_name_coverage':float(covered.mean()),'nonzero_player_seasons':int(panel.injury_events.fillna(0).gt(0).sum())}
        ix=panel.set_index(['player_id','season_start']); rows=[]
        for event in events.to_dict('records'):
            pre=(event['player_id'],event['pre_season']); post=(event['player_id'],event['post_season'])
            if pre not in ix.index or post not in ix.index: continue
            a=ix.loc[pre]; c=ix.loc[post]
            if isinstance(a,pd.DataFrame): a=a.iloc[0]
            if isinstance(c,pd.DataFrame): c=c.iloc[0]
            if pd.isna(a.get('injury_events')) or pd.isna(c.get('injury_events')): continue
            row=dict(event); row['performance_player']=a.player_name
            for outcome in ['injury_events','injury_days','injury_games_missed']:
                row['pre_'+outcome]=a[outcome]; row['post_'+outcome]=c[outcome]; row['change_'+outcome]=c[outcome]-a[outcome]
            rows.append(row)
        person=pd.DataFrame(rows); summary=[]
        if not person.empty:
            for group,data in [('all',person)]+[(str(k),v) for k,v in person.groupby('remaining_group')]:
                for outcome in ['injury_events','injury_days','injury_games_missed']:
                    delta=data['change_'+outcome].dropna(); n=len(delta)
                    summary.append({'remaining_group':group,'outcome':outcome,'n':n,'mean_pre':data['pre_'+outcome].mean(),'mean_post':data['post_'+outcome].mean(),'mean_change':delta.mean(),'std_error':delta.std(ddof=1)/math.sqrt(n) if n>1 else np.nan,'p_value':stats.ttest_1samp(delta,0).pvalue if n>1 else np.nan})
        return panel,person,pd.DataFrame(summary)
    except Exception as exc:
        b.D['warnings'].append(f'Injuries unavailable: {exc!r}')
        b.D['sources']['injuries']={'status':'failed','error':repr(exc)}
        return panel,pd.DataFrame(),pd.DataFrame()

b.add_sofascore=add_sofascore
b.regress=regress
b.injuries=injuries
b.main()
