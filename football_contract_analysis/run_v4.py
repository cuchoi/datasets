"""Fourth pass: rerun the injury section with numeric-safe days/games parsing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Executes the complete core analysis and v3 injury pass first.
import run_v3 as v3


def numeric_safe(series):
    direct = pd.to_numeric(series, errors='coerce')
    if direct.notna().mean() >= 0.75:
        return direct
    extracted = (
        series.astype(str)
        .str.extract(r'(-?\d+(?:[.,]\d+)?)')[0]
        .str.replace(',', '.', regex=False)
    )
    return pd.to_numeric(extracted, errors='coerce')


v3.numeric_from_text = numeric_safe
out = Path(__file__).resolve().parent / 'output'
panel = pd.read_csv(out / 'analysis_panel.csv', low_memory=False)
events = pd.read_csv(out / 'extension_events.csv')

# Remove the buggy v3 injury columns before reattaching corrected values.
drop_columns = [
    col for col in panel.columns
    if 'injury' in col.lower() or col in {'name_norm_tm'}
]
panel = panel.drop(columns=drop_columns, errors='ignore')

names, aggregate, schema, source_rows = v3.load_injury_data()
panel, covered = v3.attach_injuries(panel, names, aggregate)
injury_players, injury_summary = v3.event_injury_tables(panel, events)
matched_pairs, matched_summary = v3.matched_injury_did(panel, events)

diag = json.loads((out / 'diagnostics.json').read_text())
diag['warnings'] = [
    warning for warning in diag.get('warnings', [])
    if not warning.startswith('Injury') and not warning.startswith('Injuries')
]
diag['sources']['injuries'] = {
    'status': 'ok',
    'source': 'salimt/football-datasets Transfermarkt injury histories',
    'parser': 'numeric-safe v4; preserves numeric decimals and extracts numeric portion from text',
    'source_rows_after_nonmedical_exclusions': source_rows,
    'profile_names': int(len(names)),
    'panel_name_coverage': float(covered.mean()),
    'covered_panel_rows': int(covered.sum()),
    'event_pairs': int(len(injury_players)),
    'matched_did_pairs': int(len(matched_pairs)),
    'schema': schema,
    'sanity_checks': {
        'max_player_season_injury_days': float(panel['injury_days'].max()),
        'median_nonzero_player_season_injury_days': float(panel.loc[panel['injury_days'] > 0, 'injury_days'].median()),
        'max_player_season_games_missed': float(panel['injury_games_missed'].max()),
    },
}
diag['counts']['injury_event_pairs'] = int(len(injury_players))
diag['counts']['injury_matched_did_pairs'] = int(len(matched_pairs))

panel.to_csv(out / 'analysis_panel.csv', index=False)
injury_players.to_csv(out / 'injury_pre_post.csv', index=False)
injury_summary.to_csv(out / 'injury_summary.csv', index=False)
matched_pairs.to_csv(out / 'injury_matched_did_pairs.csv', index=False)
matched_summary.to_csv(out / 'injury_matched_did_summary.csv', index=False)
(out / 'diagnostics.json').write_text(json.dumps(diag, indent=2, default=str))

print(json.dumps(diag['sources']['injuries'], indent=2))
print('\nCorrected raw injury pre/post:')
print(injury_summary.to_string(index=False))
print('\nCorrected matched injury difference-in-differences:')
print(matched_summary.to_string(index=False))
