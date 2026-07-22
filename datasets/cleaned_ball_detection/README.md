# Dataset Cleaning Report

The original Roboflow exports were not modified. Ball classes were remapped to `pickleball=0`,
source-frame variants and exact duplicates were removed, negatives were capped, and splits
were reassigned by source clip to prevent adjacent-frame leakage.

## Result

- Images: 23007
- Splits: {'test': 2697, 'valid': 2489, 'train': 17821}
- Clip groups: 1079
- Source variants removed: 38893
- Exact duplicates removed: 0
- Negatives kept: 1294

See `cleaning_report.json` and `manifest.csv` for full provenance and counts.
