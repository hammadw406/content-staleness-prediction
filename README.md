# Content Staleness Prediction

Predicts whether content items are at risk of going stale, using a mix of raw 
platform metrics and custom-engineered features (momentum, activity normalization, 
visibility gap). Built with a Random Forest classifier, validated through 
5-fold cross-validation and precision-recall analysis to ensure the results 
are reliable and not an artifact of a single train/test split.

**Key result:** F1 score of 0.635 (cross-validated, threshold = 0.3), with 
engineered features (`impressions_change_pct`, `update_lag_ratio`) ranking 
among the top predictors — validating the feature engineering work.

## Highlights
- Clean, validated feature pipeline (caught and fixed 2 real bugs during 
  development: a `trend_direction` mismatch and a CTR scale issue)
- Baseline comparison: Logistic Regression vs Random Forest
- Threshold tuned via precision-recall tradeoff (favors recall ~0.75 over 
  precision ~0.55, prioritizing catching true positives)
- Stability confirmed via stratified 5-fold cross-validation (F1 std = 0.01)
- Redundant features identified and removed (`visibility_gap` dropped in 
  favor of `visibility_gap_log`, Spearman correlation = 1.0)

## Next steps
- Try Gradient Boosting as an alternative model
- Expand validation to additional time periods/datasets
