"""Sweetcrete data cleaning. Single source of truth for turning the raw lab
CSV into a model-ready frame. Import clean_sweetcrete() everywhere."""
import pandas as pd

PSI_PER_MPA = 145.038          # unit conversion constant
STRENGTH_MIN_MPA = 2.0         # physical floor for any cured concrete
STRENGTH_MAX_MPA = 120.0       # generous ceiling (UHPC territory); above = data error

def clean_sweetcrete(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 1) Recompute strength from load/area to repair calc-corrupted stress cells
    m = df["Area_in2"].notna() & df["Max_Load_lbf"].notna()
    recomputed = (df.loc[m, "Max_Load_lbf"] / df.loc[m, "Area_in2"]) / PSI_PER_MPA
    disagree = (df.loc[m, "Strength_MPa"] - recomputed).abs() / recomputed > 0.05
    idx = df.index[m][disagree.values]
    df.loc[idx, "Strength_MPa"] = recomputed[disagree.values]
    # 2) Drop physically impossible / internally inconsistent rows
    df = df[df["Strength_MPa"].between(STRENGTH_MIN_MPA, STRENGTH_MAX_MPA)].copy()
    return df.reset_index(drop=True)

# The 5 selected, deployment-valid, game-theory-justified features:
SELECTED_FEATURES = ["Age_days", "Cement_lbs", "PCC_Fraction",
                     "WaterCement_Ratio", "CoarseAgg_lbs"]
TARGET = "Strength_MPa"

if __name__ == "__main__":
    d = clean_sweetcrete("data/raw/Sweetcrete_Master_Final.csv")
    d.to_csv("data/processed/sweetcrete_clean.csv", index=False)
    print(f"Saved {len(d)} clean rows -> data/processed/sweetcrete_clean.csv")
