"""gametheory-sweetcrete — interactive GUI.

Run from the repo root with:  streamlit run app/streamlit_app.py

Three things in one app:
  1. Predict compressive strength from a chosen mix (PCC level + curing age)
  2. Explain that prediction with SHAP (game-theoretic feature contributions)
  3. Recommend the most sustainable mix that meets a required strength
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import streamlit as st

# --- locate repo root and import project modules ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import SELECTED_FEATURES                      # noqa: E402
from optimize import (feature_row, predict_strength, pareto_front,  # noqa: E402
                      nash_solution, recommend_max_pcc)

st.set_page_config(page_title="Sweetcrete Mix Designer", page_icon="🧱", layout="wide")


@st.cache_resource
def load_model():
    pipe = joblib.load(ROOT / "models" / "best_model.joblib")
    return pipe, pipe.named_steps["scaler"], pipe.named_steps["model"]


@st.cache_data
def load_metrics():
    try:
        return json.loads((ROOT / "results" / "metrics.json").read_text())
    except Exception:
        return {}


pipe, scaler, rf = load_model()
metrics = load_metrics()

# ---------------- sidebar ----------------
st.sidebar.title("🧱 Sweetcrete Mix Designer")
st.sidebar.markdown(
    "Predict and optimize the compressive strength of **Sweetcrete** — a low-carbon "
    "concrete that replaces cement with precipitated calcium carbonate (PCC), a "
    "sugar-beet refining byproduct."
)
if metrics:
    st.sidebar.metric("Model test R²", f"{metrics.get('test_r2', 0):.3f}")
    st.sidebar.metric("Test RMSE", f"{metrics.get('test_rmse', 0):.2f} MPa")
st.sidebar.caption("Random Forest + SHAP + Nash bargaining optimization")

tab_predict, tab_optimize = st.tabs(["🔮 Predict & Explain", "⚖️ Optimize Mix"])

# ---------------- tab 1: predict ----------------
with tab_predict:
    st.header("Predict compressive strength")
    left, right = st.columns([1, 1.3])
    with left:
        pcc = st.slider("PCC replacement (%)", 0, 50, 30, step=5,
                        help="Share of cement replaced by the sugar-beet byproduct.")
        age = st.slider("Curing age (days)", 1, 90, 28)
        x = feature_row(pcc, age)
        st.caption("Recipe used (from lab data at this PCC level):")
        st.dataframe(x.T.rename(columns={0: "value"}).round(3), use_container_width=True)

    pred = float(pipe.predict(x)[0])
    with right:
        st.metric("Predicted compressive strength", f"{pred:.1f} MPa")
        # SHAP explanation for this exact mix
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(scaler.transform(x))
        base = float(np.array(explainer.expected_value).ravel()[0])
        expl = shap.Explanation(values=sv[0], base_values=base,
                                data=x.iloc[0].values, feature_names=SELECTED_FEATURES)
        st.caption("Why this prediction (SHAP — game-theoretic feature contributions):")
        fig = plt.figure()
        shap.plots.waterfall(expl, show=False)
        st.pyplot(fig, clear_figure=True)

# ---------------- tab 2: optimize ----------------
with tab_optimize:
    st.header("Find the most sustainable mix that meets your target")
    c1, c2 = st.columns(2)
    with c1:
        target = st.number_input("Required strength (MPa)", 5.0, 60.0, 30.0, step=1.0)
    with c2:
        age2 = st.slider("Curing age (days)", 1, 90, 28, key="opt_age")

    rec = recommend_max_pcc(target, age2)
    if rec.get("feasible"):
        st.success(
            f"**Use {rec['PCC_%']:.0f}% PCC** — predicted {rec['predicted_strength']:.1f} MPa "
            f"(margin +{rec['margin']:.1f} MPa above your target)."
        )
    else:
        st.error(rec.get("message", "No tested PCC level meets that target at this age."))

    nash = nash_solution(age2)
    st.info(f"Balanced default (Nash bargaining solution): **{nash['PCC_%']:.0f}% PCC** "
            f"→ {nash['strength']:.1f} MPa.")

    # Pareto frontier with the target line
    d = pareto_front(age2)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(d["PCC_%"], d["strength"], s=50, color="#c0c0c0", label="all tested mixes")
    pf = d[d["pareto"]].sort_values("PCC_%")
    ax.plot(pf["PCC_%"], pf["strength"], "-o", color="#1b7837", lw=2, label="Pareto frontier")
    ax.axhline(target, color="#d7191c", ls="--", lw=1.3, label=f"target {target:.0f} MPa")
    ax.scatter([nash["PCC_%"]], [nash["strength"]], marker="*", s=350,
               color="#762a83", edgecolor="white", zorder=5, label="Nash solution")
    ax.set_xlabel("PCC replacement (%)  →  more sustainable")
    ax.set_ylabel(f"Predicted strength at {age2} d (MPa)")
    ax.set_title("Sustainability vs performance trade-off")
    ax.legend(fontsize=8)
    st.pyplot(fig, clear_figure=True)

st.caption("gametheory-sweetcrete · model predicts within the tested range "
           "(PCC 0–50%, age 1–90 d). Not validated for extrapolation.")
