# Research Log — gametheory-sweetcrete

Dated decisions, results, and dead-ends. Newest entries at the bottom.

## Step 1 — Data cleaning + feature selection
- Found and fixed a corrupted strength value (506 MPa) and dropped one inconsistent row → 178 rows.
- Used SHAP (game theory) to select 5 deployment-valid features; excluded specimen measurements
  (density/weight/dimensions) as leakage.
- Decision: regression on `Strength_MPa`; target R² ≥ 0.93.

## Step 2 — EDA
- Confirmed classic curing curve (steep gain to 28 d, then plateau) and the PCC strength penalty.
- 28-day strength holds well to ~20–30% PCC, then drops past 40%.

## Step 3 — Preprocessing
- Stratified split by PCC level; leak-proof scaling inside pipeline; 5-fold × 3 repeated CV.
- Honest baselines: Ridge R² ≈ 0.58 (weak → non-linear), Random Forest ≈ 0.88.

## Step 4 — Modeling
- Compared 8 models; tree/boosting won. Tuned RF/GB/XGB.
- Final: tuned Random Forest — test R² = 0.966, RMSE 2.45, MAE 1.54; CV R² ≈ 0.89.

## Step 5 — Interpretability
- SHAP on the trained model: age ↑ strength; PCC and w/c ↓ strength (Abrams' law). Model is
  physically sensible, not a black box.

## Step 6 — Game-theoretic optimization
- Pareto frontier of sustainability vs performance; Nash bargaining solution ≈ 30% PCC.
- Clear strength cliff past ~35% PCC.

## Step 7 — GUI
- Streamlit app: predict + live SHAP explanation + optimizer/recommender.
