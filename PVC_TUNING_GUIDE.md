# PVC Detection Parameter Tuning Guide

## Problem
Currently, PVC detection is tuned well for sample **208** but underperforms on samples **200** and **223**. The issue stems from:

1. **Fixed thresholds**: The AND rule requires BOTH premature AND wide conditions to match
2. **Patient-specific variability**: Different patients have different RR intervals and QRS morphologies
3. **No adaptive parameters**: Fixed thresholds don't generalize across different ECG recordings

## New Solutions

### 1. Diagnostic Analysis Tool
Run this to understand what's happening with each sample:

```bash
python src/analyze_pvc_performance.py --database mitdb --output mitdb_analysis
```

**Output columns:**
- `pvc_candidates`: beats matching current AND rule (premature AND wide)
- `candidates_prem_only`: beats that are ONLY premature 
- `candidates_wide_only`: beats that are ONLY wide
- `candidates_prem_or_wide`: beats matching looser OR rule
- `prem_idx_median`, `qrs_width_median_ms`: baseline stats per sample

**What to look for:**
- If `pvc_candidates` << `ref_pvc`: the AND rule is too strict
- If `candidates_prem_or_wide` > `ref_pvc`: the OR rule might be too loose
- Compare baseline statistics across samples to understand morphology differences

### 2. Alternative Detection Rules
Three detection strategies are now available via `--detection-rule`:

#### a) "and" (default, strict)
```bash
python -m src.run_mitdb --database mitdb --evaluation-mode pvc --detection-rule and
```
Requires: `premature_index < threshold` AND `qrs_width_ms > threshold`

**Best for:** Low false positive rate, but may miss PVCs

#### b) "or" (loose)
```bash
python -m src.run_mitdb --database mitdb --evaluation-mode pvc --detection-rule or
```
Requires: `premature_index < threshold` OR `qrs_width_ms > threshold`

**Best for:** Higher recall, catches more PVCs but may have more false positives

#### c) "weighted" (probabilistic)
```bash
python -m src.run_mitdb --database mitdb --evaluation-mode pvc --detection-rule weighted
```
Uses weighted scoring: `0.6 * prematurity_score + 0.4 * qrs_score > 0.5`

**Best for:** Balanced approach, adapts smoothly to intermediate cases

### 3. Parameter Tuning Strategy

#### Step 1: Diagnose the problem
```bash
python src/analyze_pvc_performance.py --database mitdb --output mitdb_analysis
# Review mitdb_analysis/pvc_analysis.csv
```

#### Step 2: Try different rules
```bash
# See if OR rule helps sample 200 and 223
python -m src.run_mitdb --database mitdb --evaluation-mode pvc --detection-rule or

# Or try weighted scoring
python -m src.run_mitdb --database mitdb --evaluation-mode pvc --detection-rule weighted
```

#### Step 3: Adjust thresholds if needed
Lower thresholds to catch more PVCs:

```bash
# Reduce prematurity threshold (was 0.95, try 0.85)
python -m src.run_mitdb --database mitdb --evaluation-mode pvc \
  --prematurity-threshold 0.85 --qrs-width-threshold-ms 85

# Or with OR rule
python -m src.run_mitdb --database mitdb --evaluation-mode pvc --detection-rule or \
  --prematurity-threshold 0.85 --qrs-width-threshold-ms 85
```

### 4. Understanding the Metrics

When comparing results:
- **Sensitivity**: % of true PVCs correctly detected (high is good)
- **Specificity**: % of non-PVCs correctly identified (high is good)
- **PPV**: % of detected beats that are actually PVCs (high is good)
- **F1**: harmonic mean of Sensitivity and PPV (balanced metric)

### 5. Sample-Specific Tuning (Advanced)

If different samples need different parameters:

```bash
# For sample 200: try looser detection
python -m src.run_mitdb --database mitdb --evaluation-mode pvc --detection-rule or

# View results in mitdb_results/pvc_eval.csv to see per-record performance
```

## Key Insights

### Why "or" might help
- If sample 200 has **narrow QRS but early beats**: OR rule will catch them
- If sample 223 has **wide QRS but late beats**: OR rule will catch them
- Trade-off: more false positives possible

### Why "weighted" might help
- Intermediate cases (moderately premature OR moderately wide)
- Smoother decision boundary than strict AND
- Produces `pvc_score` (0.0-1.0) for ranking confidence

### When to use each
- **AND**: Requires high confidence, low false positives acceptable
- **OR**: Need to catch all PVCs, false positives acceptable  
- **Weighted**: Balanced approach, want confidence ranking

## Implementation Details

### New CSV columns
- `pvc_score`: confidence score (0.0-1.0) for each beat
- Existing columns unchanged for backward compatibility

### Modified files
- [src/extract_features.py](src/extract_features.py): Added `compute_pvc_rule()` function, new `detection_rule` parameter
- [src/run_mitdb.py](src/run_mitdb.py): Added `--detection-rule` CLI argument
- [src/analyze_pvc_performance.py](src/analyze_pvc_performance.py): New diagnostic tool

## Next Steps

1. **Run diagnostic analysis** to understand baseline differences
2. **Test all three rules** on your samples
3. **Compare F1 scores** across samples
4. **Select rule** that gives best balance for your use case
5. **Fine-tune thresholds** if needed
