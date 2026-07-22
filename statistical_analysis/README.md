# Statistical Analysis

Significance testing and inter-judge agreement analysis for
*"Empathy Editing for Italian Medical Question Answering: Can Empathy Be Added
Safely?"* (CLiC-it 2026).

The script reproduces the statistics reported in the paper directly from the
per-row judgement outputs in `../IMB Generation Judgement Results/`.

## Contents
- `statistical_analysis.py` — one reproducible script (bootstrap seed 42).
  Reads the per-row CSVs from the parent repository and writes every table and
  figure below.
- `outputs/` — result tables (CSV) and `analysis_report.json`.
- `figures/` — publication figures (PDF) and companion PNGs.

## What it computes
- **McNemar's paired test** on the two judges scoring identical generated texts.
- **Bootstrap 95% confidence intervals** on the factual-pass and Safe Quality
  Improvement rates (Table 6).
- **Wilson confidence intervals** on the judge-calibration accuracies (Table 3).
- **Logistic regression** testing the length to quality-preference relationship.
- **Category-level factual-pass rates** with Wilson intervals.
- **Cohen's kappa** for inter-judge agreement on IMB-QA and for judge–gold
  agreement on the calibration sets.

## Reproduce
```
cd statistical_analysis
python statistical_analysis.py
```
Requires `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, and
`matplotlib`.
