"""Shared helpers for the Streamlit app.

The app is a thin, read-only front end on top of the `f1ml` backend package:
  * configuration / split / class labels  -> f1ml.config
  * data loading                          -> f1ml.data
  * 20 leakage-safe features              -> f1ml.features (PreRaceFeatureBuilder)
  * final saved pipeline                  -> models/final_pipeline.joblib
  * saved evaluation artefacts            -> reports/results/*.csv|json, models/model_card.json
Nothing is ever retrained here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:               # the pickled pipeline references `f1ml.*`
    sys.path.insert(0, str(SRC))

from f1ml import config, data as f1data    # noqa: E402
from f1ml.features import RAW_COLUMNS      # noqa: E402

CLASS_ORDER = config.CLASS_ORDER
TOP3, POINTS, OUTSIDE = config.TOP3, config.POINTS, config.OUTSIDE
TARGET = config.TARGET
RED = "#E10600"
CLASS_COLORS = {TOP3: "#3D85D1", POINTS: "#F2A33A", OUTSIDE: "#7A7F8C"}
MODEL_COLORS = {
    "Baseline: majority class": "#4B5563", "Baseline: grid rule": "#9AA0A6",
    "Baseline: grid-only LogReg": "#6B7280", "Logistic Regression": "#3D85D1",
    "KNN": "#1F9E5E", "SVM (RBF)": "#9A63E0", "Random Forest": RED,
}
RES = config.RES_DIR
FIG = config.FIG_DIR

# Human-readable description + family for each of the 20 model inputs
FEATURE_INFO = {
    "grid": ("Starting grid position (after penalties)", "Qualifying & grid"),
    "quali_position": ("Qualifying classification", "Qualifying & grid"),
    "quali_gap_to_pole_pct": ("Best quali time vs pole, in %", "Qualifying & grid"),
    "grid_penalty_places": ("Grid minus quali position (places lost to penalties)", "Qualifying & grid"),
    "pit_lane_start": ("Starts from the pit lane (1 = yes)", "Qualifying & grid"),
    "season": ("Season (captures regulation eras)", "Context"),
    "driver_age": ("Driver age on race day (years)", "Context"),
    "driver_avg_finish_last5": ("Driver's mean finish, previous 5 races", "Recent form"),
    "driver_prev_finish": ("Driver's finish in the previous race", "Recent form"),
    "driver_dnf_rate_last5": ("Driver's DNF rate, previous 5 races", "Recent form"),
    "driver_avg_quali_last5": ("Driver's mean quali position, previous 5", "Recent form"),
    "constructor_points_last5": ("Team points over previous 5 weekends", "Recent form"),
    "teammate_prev_finish": ("Teammate's finish in the previous race", "Recent form"),
    "driver_avg_finish_at_circuit": ("Driver's mean finish at this circuit (earlier visits)", "Circuit & experience"),
    "driver_circuit_appearances": ("Earlier starts at this circuit", "Circuit & experience"),
    "driver_career_races": ("Career races before this one", "Circuit & experience"),
    "driver_standing_before": ("Drivers' championship rank before the race", "Championship"),
    "driver_points_before": ("Driver's points before the race", "Championship"),
    "constructor_standing_before": ("Constructors' championship rank before the race", "Championship"),
    "constructor_points_before": ("Team points before the race", "Championship"),
}

CSS = f"""
<style>
    .block-container {{padding-top: 2rem; max-width: 1280px;}}
    h1, h2, h3 {{font-weight: 800; letter-spacing: -0.01em;}}
    h1 {{border-left: 6px solid {RED}; padding-left: 1rem;}}
    .f1-card {{background:#1F1F2B; border:1px solid #2E2E3A; border-radius:10px;
              padding:1rem 1.1rem; height:100%;}}
    .f1-card .label {{color:#A0A0B0; font-size:.78rem; text-transform:uppercase; letter-spacing:.06em;}}
    .f1-card .value {{color:#F2F2F5; font-size:1.8rem; font-weight:800; line-height:1.2;}}
    .f1-card .sub {{color:#A0A0B0; font-size:.85rem;}}
    .f1-stripe {{height:4px; background:linear-gradient(90deg,{RED} 0%,{RED} 40%,transparent 40%);
                margin:.2rem 0 1.2rem 0;}}
    .pill {{display:inline-block; padding:.15rem .7rem; border-radius:999px; font-weight:700; font-size:.9rem;}}
    .step {{background:#1F1F2B; border:1px solid #2E2E3A; border-left:4px solid {RED}; border-radius:8px;
           padding:.7rem .9rem; height:100%;}}
    .step b {{color:#F2F2F5;}} .step span {{color:#A0A0B0; font-size:.85rem;}}
    div[data-testid="stMetricValue"] {{font-weight:800;}}
</style>
"""


def setup_page(title: str, icon: str = "🏁"):
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)


def card(label: str, value: str, sub: str = ""):
    st.markdown(f'<div class="f1-card"><div class="label">{label}</div><div class="value">{value}</div>'
                f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def pill(category: str) -> str:
    return (f'<span class="pill" style="background:{CLASS_COLORS.get(category, "#888")};color:#fff">'
            f'{category}</span>')


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=80, b=10),
                      font=dict(family="Helvetica, Arial, sans-serif", size=13),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left", title=None),
                      hoverlabel=dict(bgcolor="#1F1F2B"))
    fig.update_xaxes(gridcolor="#2E2E3A", zeroline=False)
    fig.update_yaxes(gridcolor="#2E2E3A", zeroline=False)
    return fig


# --------------------------------------------------------------------------- cached loaders
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return f1data.load_data()


@st.cache_data(show_spinner=False)
def load_csv(name: str, index_col=None) -> pd.DataFrame | None:
    p = RES / name
    return pd.read_csv(p, index_col=index_col) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_json(path_str: str):
    p = Path(path_str)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


META_COLS = ["race_id", "driver_id", "round", "race_date", "circuit_name", "constructor_name",
             "finish_position", "quali_position"]


def _with_meta(frame: pd.DataFrame) -> pd.DataFrame:
    meta = load_dataset()[META_COLS]
    drop = [c for c in META_COLS[2:] if c in frame.columns]
    out = frame.drop(columns=drop).merge(meta, on=["race_id", "driver_id"], how="left")
    out["actual_result"] = np.where(out["finish_position"].notna(),
                                    "P" + out["finish_position"].fillna(0).astype(int).astype(str),
                                    out["finish_status"])
    return out


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame:
    """Saved test-set (2024-25) predictions of the final pipeline + probability of the actual outcome."""
    t = _with_meta(pd.read_csv(RES / "test_predictions.csv"))
    return add_surprise(t, actual_col=TARGET, pred_col="predicted")


@st.cache_data(show_spinner=False)
def load_validation_predictions() -> pd.DataFrame | None:
    """2022-23 predictions of the frozen reference RF (reproduced exactly by the final pipeline)."""
    p = config.FROZEN_DIR / "validation_predictions.csv"
    if not p.exists():
        return None
    v = pd.read_csv(p).rename(columns={
        "pred_reference_rf": "predicted",
        **{f"reference_rf_probability_{c}": f"proba_{c}" for c in CLASS_ORDER}})
    v = _with_meta(v.merge(load_dataset()[["race_id", "driver_id", "race_name", "driver_name", "grid",
                                           "finish_status"]], on=["race_id", "driver_id"], how="left"))
    return add_surprise(v, actual_col=TARGET, pred_col="predicted")


def add_surprise(frame: pd.DataFrame, actual_col: str, pred_col: str) -> pd.DataFrame:
    f = frame.copy()
    proba = f[[f"proba_{c}" for c in CLASS_ORDER]].to_numpy()
    idx = f[actual_col].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    f["p_actual"] = proba[np.arange(len(f)), idx]
    f["surprise"] = -np.log(np.clip(f["p_actual"], 1e-6, 1))
    f["correct"] = f[actual_col] == f[pred_col]
    return f


@st.cache_resource(show_spinner="Loading trained pipeline…")
def load_pipeline():
    import joblib
    return joblib.load(config.MODELS_DIR / "final_pipeline.joblib")


def require(paths: list[Path]) -> bool:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        st.error("Required project files are missing:\n\n" +
                 "\n".join(f"- `{Path(p).relative_to(PROJECT_ROOT)}`" for p in missing) +
                 "\n\nRun the app from the project folder: `streamlit run app/streamlit_app.py`.")
        return False
    return True


# --------------------------------------------------------------------------- prediction engine
@st.cache_resource(show_spinner="Building pre-race features for every race (one-off)…")
def all_race_features() -> pd.DataFrame:
    """The 20 model inputs for every driver-race, from the pipeline's own PreRaceFeatureBuilder.

    Features are leakage-safe by construction (strictly earlier races only; tests/test_no_leakage.py shows a
    race's own outcome never changes its features), so computing them once for the whole table gives the
    same values as featurising each race as 'not yet run' -- the app verifies this against the saved test
    predictions (max |diff| ~ 1e-16).
    """
    df = load_dataset()
    builder = load_pipeline().named_steps["features"]
    feats = builder.transform(f1data.raw_inputs(df).reset_index(drop=True))
    feats = feats.reset_index(drop=True)
    feats.insert(0, "driver_id", df.driver_id.to_numpy())
    feats.insert(0, "race_id", df.race_id.to_numpy())
    return feats


def predict_race(race_id: str, overrides: dict[str, dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict every driver of `race_id` with the saved pipeline (imputer -> scaler -> Random Forest).

    `overrides` = {driver_id: {"grid": 3, "quali_position": 2, "quali_gap_to_pole_pct": 0.4,
    "pit_lane_start": 0}} answers what-if questions: those raw inputs are changed and the dependent
    feature `grid_penalty_places` (= grid - quali_position) is recomputed.
    Returns (predictions, model-input features).
    """
    pipe = load_pipeline()
    df = load_dataset()
    target = df[df.race_id == race_id].reset_index(drop=True)
    feats = all_race_features()
    feats = feats[feats.race_id == race_id].reset_index(drop=True)
    assert (feats.driver_id == target.driver_id).all()
    X = feats[config.FEATURES].copy()
    for drv, vals in (overrides or {}).items():
        m = (target.driver_id == drv).to_numpy()
        for col, val in vals.items():
            X.loc[m, col] = val
        X.loc[m, "grid_penalty_places"] = X.loc[m, "grid"] - X.loc[m, "quali_position"]
    proba = pipe[1:].predict_proba(X)               # prep -> classifier (classes_ sorted alphabetically)
    out = target[["race_id", "season", "round", "race_name", "circuit_name", "driver_id", "driver_name",
                  "constructor_name", "finish_position", "finish_status", TARGET]].copy()
    out["grid"], out["quali_position"] = X["grid"].to_numpy(), X["quali_position"].to_numpy()
    for i, c in enumerate(pipe.classes_):
        out[f"proba_{c}"] = proba[:, i]
    out["predicted"] = pipe.classes_[proba.argmax(axis=1)]
    return out, X.assign(driver_id=target.driver_id.to_numpy())      # (predictions, the 20 inputs used)
