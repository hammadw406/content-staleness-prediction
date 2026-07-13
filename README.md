# Content Staleness Prediction

Predicts which web pages need a content refresh first, using a mix of raw
search-performance metrics and custom-engineered features (momentum,
normalized activity, staleness, opportunity/visibility signals). Built with
a Random Forest classifier, validated through 5-fold cross-validation and a
full precision-recall analysis to confirm the results are reliable and not
an artifact of a single train/test split.

**Key result:** F1 score of **0.635** (cross-validated, threshold = 0.3),
with engineered features (`impressions_change_pct`, `update_lag_ratio`)
ranking among the top predictors — validating the feature engineering work.

## Highlights

- Clean, validated feature engineering pipeline — caught and fixed 2 real
  bugs during development: a `trend_direction` category mismatch and a CTR
  scale issue (values on a 0–100 scale mixed in with 0–1)
- Baseline comparison: Logistic Regression vs. Random Forest
- Threshold tuned via the full precision-recall tradeoff (deliberately
  favors recall ~0.75 over precision ~0.55, prioritizing catching pages
  that actually need a refresh)
- Result stability confirmed via stratified 5-fold cross-validation
  (F1 std = 0.01 across folds)
- Redundant features identified and resolved (`visibility_gap` dropped in
  favor of `visibility_gap_log`, confirmed via Spearman correlation = 1.0)

## Data

`content_refresh_anonymized.csv` — ~30k web pages with search performance
data (rankings, clicks, impressions, engagement, content metadata). Not
included in this repo; place it in the project root to run the notebook.

## Project structure

```
.
├── content_refresh_prediction.ipynb   # full pipeline: EDA → feature engineering →
│                                       # modeling → validation → write-up
└── README.md
```

## Pipeline overview

1. Data exploration & missing value handling
2. Type casting
3. Outlier detection (domain-rule based)
4. Feature engineering (momentum, activity, staleness, opportunity features)
5. Label definition (`needs_refresh`, with justified iteration)
6. Encoding, leakage-column removal, train/test split
7. Baseline models: Logistic Regression vs. Random Forest
8. Threshold tuning
9. Feature importance analysis
10. Gradient Boosting (exploratory comparison)
11. Cross-validation (stability check)
12. Precision-recall curve
13. Feature redundancy check & cleanup

## Next steps

- Tune Gradient Boosting properly (only run with defaults so far) and
  compare against the Random Forest baseline
- Validate the pipeline against additional time periods or datasets

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook content_refresh_prediction.ipynb
```
