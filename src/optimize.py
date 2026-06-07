"""Step 6 — game-theoretic mix optimization.

Frames Sweetcrete mix design as a two-objective game:
  * Sustainability  -> maximize PCC replacement (displaces cement, uses byproduct)
  * Performance     -> maximize predicted compressive strength

The trained model (models/best_model.joblib) is the objective function. We expose the
Pareto-efficient frontier, the Nash bargaining solution (the fair compromise), and a
constraint-based recommender ("max PCC that still meets a required strength").

Mix proportions are tied to the PCC level in the experimental design, so the decision
variable is the PCC replacement level (a tested level: 0,5,...,50%), and we use the
representative recipe measured at that level. Staying on tested levels avoids extrapolation.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
from data_cleaning import clean_sweetcrete, SELECTED_FEATURES  # noqa: E402

_MODEL = None
_RECIPE = None


def _load():
    global _MODEL, _RECIPE
    if _MODEL is None:
        _MODEL = joblib.load(ROOT / "models" / "best_model.joblib")
        df = clean_sweetcrete(str(ROOT / "data" / "raw" / "Sweetcrete_Master_Final.csv"))
        _RECIPE = (df.groupby("PCC_%")[["PCC_Fraction", "Cement_lbs",
                                        "WaterCement_Ratio", "CoarseAgg_lbs"]]
                     .median().reset_index())
    return _MODEL, _RECIPE


def predict_strength(pcc_level: float, age_days: float) -> float:
    """Predicted compressive strength (MPa) for a tested PCC level at a given age."""
    model, recipe = _load()
    r = recipe.iloc[(recipe["PCC_%"] - pcc_level).abs().argmin()]
    x = pd.DataFrame([{"Age_days": age_days, "Cement_lbs": r["Cement_lbs"],
                       "PCC_Fraction": r["PCC_Fraction"],
                       "WaterCement_Ratio": r["WaterCement_Ratio"],
                       "CoarseAgg_lbs": r["CoarseAgg_lbs"]}])[SELECTED_FEATURES]
    return float(model.predict(x)[0])


def strength_curve(age_days: float = 28) -> pd.DataFrame:
    """Predicted strength at every tested PCC level for a given age."""
    _, recipe = _load()
    out = [{"PCC_%": lvl, "strength": predict_strength(lvl, age_days)}
           for lvl in recipe["PCC_%"]]
    return pd.DataFrame(out)


def pareto_front(age_days: float = 28) -> pd.DataFrame:
    """Non-dominated points maximizing both PCC and strength."""
    d = strength_curve(age_days).sort_values("PCC_%").reset_index(drop=True)
    keep = []
    for i, row in d.iterrows():
        dominated = ((d["PCC_%"] > row["PCC_%"]) & (d["strength"] >= row["strength"])).any()
        keep.append(not dominated)
    d["pareto"] = keep
    return d


def nash_solution(age_days: float = 28) -> dict:
    """Nash bargaining solution: maximize product of normalized objectives."""
    d = strength_curve(age_days)
    p = d["PCC_%"].values / d["PCC_%"].max()
    s = d["strength"].values - d["strength"].min()
    s = s / s.max() if s.max() > 0 else s
    i = int(np.argmax(p * s))
    return {"PCC_%": float(d["PCC_%"].iloc[i]),
            "strength": float(d["strength"].iloc[i]), "age_days": age_days}


def recommend_max_pcc(target_strength: float, age_days: float = 28) -> dict:
    """Most sustainable mix (max PCC) whose predicted strength meets the target."""
    d = strength_curve(age_days)
    ok = d[d["strength"] >= target_strength]
    if ok.empty:
        return {"feasible": False, "message": "No tested PCC level meets that target."}
    best = ok.loc[ok["PCC_%"].idxmax()]
    return {"feasible": True, "PCC_%": float(best["PCC_%"]),
            "predicted_strength": float(best["strength"]),
            "margin": float(best["strength"] - target_strength), "age_days": age_days}


if __name__ == "__main__":
    print("Nash solution (28d):", nash_solution())
    print("Need 30 MPa @28d ->", recommend_max_pcc(30, 28))
