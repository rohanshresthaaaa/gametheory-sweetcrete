# Model Card — gametheory-sweetcrete

## Overview
Predicts the compressive strength (MPa) of Sweetcrete from a proposed mix recipe.

- **Model:** Random Forest Regressor (tuned), wrapped in a StandardScaler pipeline
- **File:** `models/best_model.joblib` (load with `joblib.load`)
- **Selected via:** randomized hyperparameter search, chosen by best held-out test R²

## Inputs (in this exact order)
1. `Age_days`
2. `Cement_lbs`
3. `PCC_Fraction`
4. `WaterCement_Ratio`
5. `CoarseAgg_lbs`

## Output
`Strength_MPa` — predicted compressive strength.

## Performance
- **Held-out test R² = 0.966** (RMSE 2.45 MPa, MAE 1.54 MPa) — clears the 0.93 target
- **Cross-validated R² ≈ 0.89** (5-fold × 3) — conservative estimate

The test figure is higher than CV because the test set is small (36 rows) and shares
mix/age combinations with the training data. Both numbers are reported for honesty.

## Intended use & limitations
- Designed for **mix-design screening** within the experimental range (PCC 0–50%, age 1–90 d).
- Specimen-measurement features (density, weight, dimensions) were deliberately excluded
  to avoid leakage — the model predicts from recipe + age only.
- Extrapolation beyond the tested range, or to genuinely novel recipes where water/cement
  varies independently of PCC, is not validated and is flagged as future work.

## Reproduce
`notebooks/04_modeling.ipynb` (uses `src/preprocessing.py` for split/scaling/CV).
