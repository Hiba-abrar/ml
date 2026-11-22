"""Load the saved pipeline and predict the finishing category for existing races.

    python predict_example.py            # predicts the 2025 season opener as a demo

For a NEW race, append the completed races since 2025 to the history first:
    pipe["features"].update_history(completed_rows)       # raw columns, outcomes known
    pipe.predict(upcoming_entries)                         # same raw columns, outcome columns NaN
Raw columns required: see f1ml.features.RAW_COLUMNS.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import joblib

from f1ml import config, data

pipe = joblib.load(config.MODELS_DIR / "final_pipeline.joblib")
df = data.load_data()
race = df[df.race_id == "2025_01"]
X = data.raw_inputs(race).copy()
X[["position_order", "is_dnf", "weekend_points"]] = float("nan")   # pretend the race has not happened
out = race[["driver_name", "grid"]].assign(predicted=pipe.predict(X))
proba = pipe.predict_proba(X)
for i, c in enumerate(pipe.classes_):
    out["p_" + c] = proba[:, i].round(3)
print(out.sort_values("grid").to_string(index=False))
