"""Report paired mission-resilience gains from the archived 90-run study."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/v2_2_clearance/g6/V21_G6_sequential_recovery_N8/run_records.csv'


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit('Run python tools/restore_archived_runs.py first.')
    with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
        records = list(csv.DictReader(handle))
    results = {}
    for layout in sorted({row['layout'] for row in records}):
        values = [100.0 * float(row['R_mission_gain']) / float(row['R_mission_no_recovery'])
                  for row in records if row['layout'] == layout]
        results[layout] = {'paired_runs': len(values), 'mean_relative_gain_percent': mean(values)}
    print(json.dumps({'baseline': 'matched no-recovery run', 'mission_horizon_s': 50,
                      'by_layout': results,
                      'unweighted_mean_over_layouts_percent': mean(
                          item['mean_relative_gain_percent'] for item in results.values())}, indent=2))


if __name__ == '__main__':
    main()
