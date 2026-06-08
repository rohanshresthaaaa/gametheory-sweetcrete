# Methodology — gametheory-sweetcrete

End-to-end method for predicting and optimizing the compressive strength of Sweetcrete,
a low-carbon concrete that replaces a portion of Portland cement with precipitated
calcium carbonate (PCC), a byproduct of sugar-beet refining.

## 1. Data
- **Source:** laboratory compression tests (`data/raw/Sweetcrete_Master_Final.csv`).
- **Design:** a grid of PCC replacement level (0–50% in 5% steps) × curing age
  (1, 7, 28, 56, 90 days), with replicate specimens per combination.
- **Target:** `Strength_MPa` (compressive strength).

## 2. Cleaning (`src/data_cleaning.py`)
Two data-quality issues were corrected, yielding **178 valid rows** (2.2–52.6 MPa):
1. **One corrupted strength value** (~506 MPa, physically impossible) was traced to a bad
   stress cell and recomputed from load ÷ area (→ ~37.5 MPa, physically sensible).
2. **One internally inconsistent row** (load disagreeing with stress) was removed.

Physically impossible strengths (outside 2–120 MPa) are filtered as a guard.

## 3. Feature selection (game theory — SHAP)
Shapley values (cooperative game theory) ranked candidate features. Five were selected,
balancing the SHAP ranking, non-redundancy, and **deployment validity**:

`Age_days`, `Cement_lbs`, `PCC_Fraction`, `WaterCement_Ratio`, `CoarseAgg_lbs`.

**Deliberately excluded:** specimen measurements (density, weight, cylinder dimensions).
These are known only *after* casting and curing, so using them to predict strength from a
*recipe* would be data leakage. This keeps the model usable as a true design tool.

## 4. Preprocessing (`src/preprocessing.py`)
- **Stratified train/test split** (80/20, stratified by PCC level) so every replacement
  level appears in both partitions. 142 train / 36 test.
- **Standardization** wrapped inside a scikit-learn `Pipeline`, so the scaler is re-fit on
  each fold's training data only — no leakage.
- **Repeated k-fold cross-validation** (5-fold × 3) for stable estimates on a small dataset.

## 5. Modeling (`notebooks/04_modeling.ipynb`)
Eight models were compared under the same CV; tree/boosting models dominated. The top three
(Random Forest, Gradient Boosting, XGBoost) were tuned by randomized search.

**Final model: tuned Random Forest.**

| Metric | Value |
|---|---|
| Test R² | 0.966 |
| Test RMSE | 2.45 MPa |
| Test MAE | 1.54 MPa |
| Cross-validated R² | ≈ 0.89 |

The test R² exceeds the CV R² because the test set is small (36 rows) and shares mix/age
combinations with training. **Both are reported** for honesty; the CV figure is the
conservative claim.

## 6. Interpretability (game theory — SHAP) (`notebooks/05_interpretability.ipynb`)
SHAP applied to the trained model confirms it learned real concrete physics: higher curing
age raises strength; higher PCC replacement and higher water/cement ratio lower it (Abrams'
law). Global (beeswarm), dependence, and local (waterfall) explanations are provided.

## 7. Game-theoretic optimization (`src/optimize.py`, `notebooks/06_optimization.ipynb`)
Mix design is framed as a two-objective game: **Sustainability** (maximize PCC) vs
**Performance** (maximize strength), with the trained model as the objective.
- The **Pareto-efficient frontier** is the set of non-dominated mixes.
- The **Nash bargaining solution** (maximizing the product of normalized gains) identifies
  **≈30% PCC** as the balanced optimum, sitting at the knee before a sharp strength cliff
  past ~35%.
- A constraint-based recommender returns the most sustainable mix meeting a required strength.

## 8. Deployment (`app/streamlit_app.py`)
A Streamlit GUI exposes prediction (with live SHAP explanation) and the optimizer for
interactive use by engineers.

## Key limitations
- In this dataset, mix proportions are confounded with PCC level (the recipe was fixed per
  level), so the independent predictive signal comes mainly from age + PCC + cement.
- The model is validated only within the tested range (PCC 0–50%, age 1–90 d); extrapolation
  is not validated.
