"""Third pass: execute the validated performance analysis, then add robust injury joins."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Importing run_v2 executes the full performance/rating analysis and writes its outputs.
import run_v2 as core  # noqa: F401
import run as b


def flexcol(columns, aliases=(), contains=()):
    normalized = {b.norm(col): col for col in columns}
    for alias in aliases:
        hit = normalized.get(b.norm(alias))
        if hit is not None:
            return hit
    for col in columns:
        name = b.norm(col)
        if any(b.norm(token) in name for token in contains):
            return col
    return None


def parse_season_start(value):
    text = str(value).strip()
    four = re.search(r'(19\d{2}|20\d{2})', text)
    if four:
        return int(four.group(1))
    two = re.search(r'(?<!\d)(\d{2})\s*[/\-]', text)
    if two:
        year = int(two.group(1))
        return 2000 + year if year < 80 else 1900 + year
    return np.nan


def numeric_from_text(series):
    return pd.to_numeric(
        series.astype(str).str.replace('.', '', regex=False).str.extract(r'(\d+)')[0],
        errors='coerce',
    )


def paired_stats(values):
    values = pd.to_numeric(values, errors='coerce').dropna()
    n = len(values)
    if n == 0:
        return {'n': 0, 'mean_change': np.nan, 'std_error': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': np.nan}
    mean = float(values.mean())
    if n == 1:
        return {'n': 1, 'mean_change': mean, 'std_error': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': np.nan}
    se = float(values.std(ddof=1) / math.sqrt(n))
    crit = float(stats.t.ppf(0.975, n - 1))
    return {
        'n': n,
        'mean_change': mean,
        'std_error': se,
        'ci_low': mean - crit * se,
        'ci_high': mean + crit * se,
        'p_value': float(stats.ttest_1samp(values, 0).pvalue),
    }


def load_injury_data():
    base = 'https://raw.githubusercontent.com/salimt/football-datasets/main/datalake/transfermarkt'
    injury_path = b.get(base + '/player_injuries/player_injuries.csv', b.RAW / 'player_injuries.csv', 100000)
    profile_path = b.get(base + '/player_profiles/player_profiles.csv', b.RAW / 'player_profiles.csv', 100000)
    injuries = pd.read_csv(injury_path, low_memory=False)
    profiles = pd.read_csv(profile_path, low_memory=False)

    profile_id = flexcol(profiles.columns, ['player_id', 'player id'], ['player id'])
    profile_name = flexcol(profiles.columns, ['player_name', 'player name', 'name'], ['player name'])
    injury_id = flexcol(injuries.columns, ['player_id', 'player id'], ['player id'])
    season_col = flexcol(injuries.columns, ['season', 'season_name'], ['season'])
    reason_col = flexcol(injuries.columns, ['injury_reason', 'injury reason', 'injury', 'reason'], ['injury reason', 'reason'])
    days_col = flexcol(injuries.columns, ['days_missed', 'days missed', 'days'], ['days missed'])
    games_col = flexcol(injuries.columns, ['games_missed', 'games missed', 'matches_missed'], ['games missed', 'matches missed'])

    schema = {
        'profile_columns': profiles.columns.tolist(),
        'injury_columns': injuries.columns.tolist(),
        'selected': {
            'profile_id': profile_id,
            'profile_name': profile_name,
            'injury_id': injury_id,
            'season': season_col,
            'reason': reason_col,
            'days': days_col,
            'games': games_col,
        },
    }
    if profile_id is None or profile_name is None or injury_id is None or season_col is None:
        raise RuntimeError('Required injury/profile columns not found: ' + json.dumps(schema))

    names = profiles[[profile_id, profile_name]].copy()
    names.columns = ['tm_player_id', 'tm_player_name']
    names['tm_player_id'] = pd.to_numeric(names['tm_player_id'], errors='coerce')
    names['name_norm_tm'] = names['tm_player_name'].map(b.norm)
    names = names.dropna(subset=['tm_player_id']).drop_duplicates('tm_player_id')

    data = injuries.copy()
    data['tm_player_id'] = pd.to_numeric(data[injury_id], errors='coerce')
    data['season_start'] = data[season_col].map(parse_season_start)
    data['injury_reason'] = data[reason_col].fillna('').astype(str) if reason_col else ''
    data['reason_norm'] = pd.Series(data['injury_reason'], index=data.index).map(b.norm)
    data['injury_days'] = numeric_from_text(data[days_col]) if days_col else np.nan
    data['injury_games_missed'] = numeric_from_text(data[games_col]) if games_col else np.nan

    non_medical = data['reason_norm'].str.contains(
        r'\brest\b|suspension|disciplin|personal reason|not in squad|international duty',
        regex=True,
        na=False,
    )
    data = data[~non_medical & data['season_start'].isin(b.SEASONS)].copy()
    data = data.merge(names, on='tm_player_id', how='left')

    agg = data.groupby(['name_norm_tm', 'season_start'], as_index=False).agg(
        injury_events=('tm_player_id', 'size'),
        injury_days=('injury_days', 'sum'),
        injury_games_missed=('injury_games_missed', 'sum'),
    )
    return names, agg, schema, len(data)


def attach_injuries(panel, names, aggregate):
    panel = panel.copy()
    panel['name_norm_injury_v3'] = panel['player_name'].map(b.norm)
    panel = panel.merge(
        aggregate,
        left_on=['name_norm_injury_v3', 'season_start'],
        right_on=['name_norm_tm', 'season_start'],
        how='left',
        suffixes=('', '_v3'),
    )
    universe = set(names['name_norm_tm'].dropna())
    covered = panel['name_norm_injury_v3'].isin(universe)
    for outcome in ['injury_events', 'injury_days', 'injury_games_missed']:
        v3 = outcome + '_v3'
        if v3 in panel:
            panel[outcome] = panel[v3]
        if outcome not in panel:
            panel[outcome] = np.nan
        panel.loc[covered, outcome] = panel.loc[covered, outcome].fillna(0.0)
    return panel, covered


def event_injury_tables(panel, events):
    outcomes = ['injury_events', 'injury_days', 'injury_games_missed']
    lookup = panel.set_index(['player_id', 'season_start'])
    rows = []
    for event in events.to_dict('records'):
        pre_key = (event['player_id'], event['pre_season'])
        post_key = (event['player_id'], event['post_season'])
        if pre_key not in lookup.index or post_key not in lookup.index:
            continue
        pre = lookup.loc[pre_key]
        post = lookup.loc[post_key]
        if isinstance(pre, pd.DataFrame):
            pre = pre.iloc[0]
        if isinstance(post, pd.DataFrame):
            post = post.iloc[0]
        if any(pd.isna(pre.get(outcome)) or pd.isna(post.get(outcome)) for outcome in outcomes):
            continue
        row = dict(event)
        row['performance_player'] = pre['player_name']
        for outcome in outcomes:
            row['pre_' + outcome] = pre[outcome]
            row['post_' + outcome] = post[outcome]
            row['change_' + outcome] = post[outcome] - pre[outcome]
        rows.append(row)
    player_level = pd.DataFrame(rows)
    summaries = []
    if not player_level.empty:
        groups = [('all', player_level)] + [(str(k), v) for k, v in player_level.groupby('remaining_group')]
        for group_name, data in groups:
            for outcome in outcomes:
                stats_row = paired_stats(data['change_' + outcome])
                summaries.append({
                    'remaining_group': group_name,
                    'outcome': outcome,
                    'mean_pre': float(data['pre_' + outcome].mean()),
                    'mean_post': float(data['post_' + outcome].mean()),
                    **stats_row,
                })
    return player_level, pd.DataFrame(summaries)


def matched_injury_did(panel, events):
    outcomes = ['injury_events', 'injury_days', 'injury_games_missed']
    lookup = panel.set_index(['player_id', 'season_start'])
    event_keys = set(zip(events['player_id'], events['pre_season']))
    pairs = []
    for event in events.to_dict('records'):
        year = int(event['pre_season'])
        treated_pre_key = (event['player_id'], year)
        treated_post_key = (event['player_id'], year + 1)
        if treated_pre_key not in lookup.index or treated_post_key not in lookup.index:
            continue
        treated_pre = lookup.loc[treated_pre_key]
        treated_post = lookup.loc[treated_post_key]
        if isinstance(treated_pre, pd.DataFrame): treated_pre = treated_pre.iloc[0]
        if isinstance(treated_post, pd.DataFrame): treated_post = treated_post.iloc[0]
        if any(pd.isna(treated_pre.get(x)) or pd.isna(treated_post.get(x)) for x in outcomes):
            continue

        candidates = panel[
            (panel['season_start'] == year)
            & (panel['league'] == event['league'])
            & (panel['position_group'] == event['position_group'])
            & (panel['remaining_group'] == event['remaining_group'])
            & (panel['player_id'] != event['player_id'])
            & panel['age'].between(treated_pre['age'] - 1.5, treated_pre['age'] + 1.5)
        ].copy()
        valid = []
        for candidate in candidates.itertuples(index=False):
            pre_key = (candidate.player_id, year)
            post_key = (candidate.player_id, year + 1)
            if post_key not in lookup.index or pre_key in event_keys:
                continue
            post = lookup.loc[post_key]
            if isinstance(post, pd.DataFrame): post = post.iloc[0]
            if b.clubnorm(candidate.club_name) != b.clubnorm(post['club_name']):
                continue
            if any(pd.isna(getattr(candidate, x)) or pd.isna(post.get(x)) for x in outcomes):
                continue
            valid.append(candidate.player_id)
        candidates = candidates[candidates['player_id'].isin(valid)].copy()
        if candidates.empty:
            continue
        features = ['age', 'minutes', 'injury_events', 'injury_days', 'injury_games_missed']
        combined = pd.concat([pd.DataFrame([treated_pre]), candidates], ignore_index=True)
        matrix = pd.DataFrame(index=combined.index)
        for feature in features:
            values = pd.to_numeric(combined[feature], errors='coerce')
            if feature in ['minutes', 'injury_days', 'injury_games_missed']:
                values = np.log1p(values)
            values = values.fillna(values.median())
            sd = values.std(ddof=0)
            matrix[feature] = (values - values.mean()) / (sd if sd > 1e-9 else 1.0)
        distances = np.sqrt(((matrix.iloc[1:].to_numpy() - matrix.iloc[0].to_numpy()) ** 2).sum(axis=1))
        chosen_pos = int(np.argmin(distances))
        control_pre = candidates.iloc[chosen_pos]
        control_post = lookup.loc[(control_pre['player_id'], year + 1)]
        if isinstance(control_post, pd.DataFrame): control_post = control_post.iloc[0]
        row = {
            'event_id': event['event_id'],
            'treated_player': treated_pre['player_name'],
            'control_player': control_pre['player_name'],
            'league': event['league'],
            'remaining_group': event['remaining_group'],
            'pre_season': year,
            'match_distance': float(distances[chosen_pos]),
        }
        for outcome in outcomes:
            treated_change = treated_post[outcome] - treated_pre[outcome]
            control_change = control_post[outcome] - control_pre[outcome]
            row['treated_change_' + outcome] = treated_change
            row['control_change_' + outcome] = control_change
            row['did_' + outcome] = treated_change - control_change
        pairs.append(row)
    pair_df = pd.DataFrame(pairs)
    summary = []
    if not pair_df.empty:
        groups = [('all', pair_df)] + [(str(k), v) for k, v in pair_df.groupby('remaining_group')]
        for group_name, data in groups:
            for outcome in outcomes:
                summary.append({
                    'remaining_group': group_name,
                    'outcome': outcome,
                    **paired_stats(data['did_' + outcome]),
                    'mean_treated_change': float(data['treated_change_' + outcome].mean()),
                    'mean_control_change': float(data['control_change_' + outcome].mean()),
                })
    return pair_df, pd.DataFrame(summary)


out = Path(__file__).resolve().parent / 'output'
panel = pd.read_csv(out / 'analysis_panel.csv', low_memory=False)
events = pd.read_csv(out / 'extension_events.csv')
diag = json.loads((out / 'diagnostics.json').read_text())

try:
    names, aggregate, schema, source_rows = load_injury_data()
    panel, covered = attach_injuries(panel, names, aggregate)
    injury_players, injury_summary = event_injury_tables(panel, events)
    matched_pairs, matched_summary = matched_injury_did(panel, events)
    diag['warnings'] = [warning for warning in diag.get('warnings', []) if not warning.startswith('Injuries unavailable')]
    diag['sources']['injuries'] = {
        'status': 'ok',
        'source': 'salimt/football-datasets Transfermarkt injury histories',
        'source_rows_after_nonmedical_exclusions': source_rows,
        'profile_names': int(len(names)),
        'panel_name_coverage': float(covered.mean()),
        'covered_panel_rows': int(covered.sum()),
        'event_pairs': int(len(injury_players)),
        'matched_did_pairs': int(len(matched_pairs)),
        'schema': schema,
    }
    panel.to_csv(out / 'analysis_panel.csv', index=False)
    injury_players.to_csv(out / 'injury_pre_post.csv', index=False)
    injury_summary.to_csv(out / 'injury_summary.csv', index=False)
    matched_pairs.to_csv(out / 'injury_matched_did_pairs.csv', index=False)
    matched_summary.to_csv(out / 'injury_matched_did_summary.csv', index=False)
except Exception as exc:
    diag.setdefault('warnings', []).append('Injury v3 failed: ' + repr(exc))
    diag['sources']['injuries_v3'] = {'status': 'failed', 'error': repr(exc)}

(out / 'diagnostics.json').write_text(json.dumps(diag, indent=2, default=str))
print(json.dumps(diag['sources'].get('injuries', diag['sources'].get('injuries_v3')), indent=2))
if (out / 'injury_summary.csv').exists() and (out / 'injury_summary.csv').stat().st_size > 1:
    print(pd.read_csv(out / 'injury_summary.csv').to_string(index=False))
if (out / 'injury_matched_did_summary.csv').exists() and (out / 'injury_matched_did_summary.csv').stat().st_size > 1:
    print(pd.read_csv(out / 'injury_matched_did_summary.csv').to_string(index=False))
