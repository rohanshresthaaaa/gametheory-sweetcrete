# gametheory-sweetcrete

Machine-learning prediction & game-theoretic optimization of **Sweetcrete** compressive
strength. Sweetcrete is a low-carbon concrete developed at Idaho State University that
replaces a portion of Portland cement with **Precipitated Calcium Carbonate (PCC)**, a
byproduct of sugar-beet refining.

## Mission
Build an accurate, interpretable model that predicts compressive strength (MPa) from a
**proposed mix recipe**, and wrap it in a tool engineers can use to design project-specific
Sweetcrete mixes. Target: **R^2 >= 0.93** on held-out data.

## Deliverables (tracked through the project)
- [x] Cleaned dataset + reproducible cleaning module
- [x] Game-theoretic (SHAP) feature selection
- [ ] EDA notebook
- [ ] Modeling notebook (multiple ML models, tuned, cross-validated)
- [ ] Evaluation + interpretability (SHAP) notebook
- [ ] Game-theoretic mix optimization
- [ ] GUI (Streamlit) for live mix-strength prediction
- [ ] Paper draft, poster, documentation, future-work log

## Selected features (deployment-valid, SHAP-ranked)
1. `Age_days` 2. `Cement_lbs` 3. `PCC_Fraction` 4. `WaterCement_Ratio` 5. `CoarseAgg_lbs`
Target: `Strength_MPa`. (Specimen measurements like density/weight were EXCLUDED — they
are only knowable after curing and would leak the answer.)

## Repo layout
data/raw  data/processed  src  notebooks  models  results/figures  docs

## Quickstart
pip install -r requirements.txt
python src/data_cleaning.py

## Pipeline status
Step 1 (data + features) complete. Next: Step 2 — EDA notebook.
