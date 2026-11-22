"""Overview: problem, headline results (reported honestly), pipeline, split, data sources."""
import plotly.express as px
import streamlit as st

from app_utils import (CLASS_COLORS, CLASS_ORDER, PROJECT_ROOT, RED, RES, card, config, header, load_csv,
                       load_dataset, load_json, require, setup_page, style_fig)

setup_page("Overview")
header("Grid Position vs. Race Reality",
       "Can we predict, before the lights go out, whether a Formula 1 driver finishes Top 3, in the Points, "
       "or Outside the Points?")

if not require([config.CLEAN_FILE, RES / "final_results.json", config.MODELS_DIR / "model_card.json"]):
    st.stop()

df = load_dataset()
res = load_json(str(RES / "final_results.json"))
card_json = load_json(str(config.MODELS_DIR / "model_card.json"))
sources = load_json(str(config.DATA_DIR / "data_sources.json")) or {}
test, base = res["test"], res["baseline_grid_rule_test"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    card("Test accuracy (2024–25)", f"{test['accuracy']:.1%}", f"grid-rule baseline {base['accuracy']:.1%}")
with c2:
    card("Test macro-F1", f"{test['macro_f1']:.3f}", f"grid-rule baseline {base['macro_f1']:.3f}")
with c3:
    card("Test ROC-AUC (OvR)", f"{test['roc_auc_ovr_macro']:.3f}", "probabilities rank outcomes well")
with c4:
    card("5-fold CV macro-F1", f"{res['final_rf_cv_macro_f1_mean']:.3f}",
         f"± {res['final_rf_cv_macro_f1_std']:.3f} (train seasons)")

diff = (base["accuracy"] - test["accuracy"]) * 100
st.write("")
st.warning(
    f"**Reported as found:** on the untouched 2024–25 test seasons the trivial *“finish where you start”* rule "
    f"scores **{diff:.1f} points higher** in accuracy than the Random Forest, even though the forest beat it by "
    f"~6 points on 2022–23 validation. The grid is an extremely strong signal; what the model adds is "
    f"a full probability for each category, built from context (form, standings, teammates), rather than a headline accuracy gain. "
    f"See **Test Evaluation**.")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("The task")
    st.markdown(
        """
Multiclass classification, one row per **driver per race** (2010–2025). Using only information available
**before the race starts**, predict the finishing category:

| Category | Definition |
|---|---|
| **Top 3** | Finished P1–P3 (podium) |
| **Points** | Finished P4–P10 |
| **Outside Points** | P11 or lower, or not classified (DNF / DSQ / DNS) |

Categories are used instead of exact positions because exact positions are very noisy, while podium / points /
no-points is what matters, and macro-F1 lets the three classes count equally.
""")
with right:
    counts = df[config.TARGET].value_counts().reindex(CLASS_ORDER).reset_index()
    counts.columns = ["category", "records"]
    fig = px.bar(counts, x="category", y="records", color="category", text="records",
                 color_discrete_map=CLASS_COLORS, title="Class balance (all 16 seasons)")
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Driver-race records")
    st.plotly_chart(style_fig(fig, 340), width="stretch")

st.subheader("What was built")
hp = card_json["hyperparameters"]
steps = [
    ("1 · PreRaceFeatureBuilder", "Raw race entries → 20 leakage-safe features (strictly earlier races only)"),
    ("2 · Imputer + indicators", "Median imputation with missing-value flags, fitted on training rows only"),
    ("3 · StandardScaler", "Standardisation, fitted on training rows only"),
    ("4 · Random Forest", f"{hp['n_estimators']} trees · depth {hp['max_depth']} · leaf {hp['min_samples_leaf']} · "
                          f"max_features {hp['max_features']}"),
]
for col, (t, d) in zip(st.columns(4), steps):
    col.markdown(f'<div class="step"><b>{t}</b><br><span>{d}</span></div>', unsafe_allow_html=True)

st.write("")
c1, c2 = st.columns([1.1, 1])
with c1:
    st.subheader("Chronological split (never shuffled)")
    split = load_csv("split_summary.csv", index_col=0)
    split = split.reset_index().rename(columns={"index": "split"})
    fig = px.bar(split, x="rows", y="split", orientation="h", color="split", text="seasons",
                 color_discrete_map={"train": "#3D85D1", "validation": "#F2A33A", "test": RED},
                 hover_data=["races", "first_race", "last_race"])
    fig.update_traces(textposition="inside")
    fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed", title=None), xaxis_title="Driver-race rows")
    st.plotly_chart(style_fig(fig, 260), width="stretch")
    st.caption("Train fits and tunes (5-fold expanding-window CV over race weekends) · validation compares models · "
               "test is reported **once** at the very end.")
with c2:
    st.subheader("Data provenance")
    f1, w = sources.get("f1_data", {}), sources.get("weather", {})
    st.markdown(
        f"**Results & qualifying:** {f1.get('source', 'n/a')}  \n"
        f"{f1.get('rows', {}).get('results', '?'):,} result rows · {f1.get('rows', {}).get('races', '?')} races  \n"
        f"Retrieved {str(f1.get('retrieval_date_utc', 'n/a'))[:10]}  \n\n"
        f"**Weather (exploration only):** {w.get('source', 'n/a')}  \n"
        f"Licence {w.get('licence', 'n/a')}")
    st.caption("The cleaned table can be rebuilt exactly from `data/interim/` with `f1ml.data_prep`.")

with st.expander("Project requirement checklist → where to see it in this app"):
    st.markdown("""
| # | Requirement | Where |
|---|---|---|
| 1 | EDA with ≥ 5 plots | **Data & EDA** (interactive) + saved notebook figures |
| 2 | Pre-processing in a Pipeline fitted on train only | **Pipeline & Features** |
| 3 | Time-ordered train / validation / test split | **Overview**, **Model Lab** |
| 4 | Baselines + ≥ 3 algorithms | **Model Lab → Comparison** |
| 5 | 5-fold CV hyper-parameter tuning | **Model Lab → Hyper-parameter search** |
| 6 | Test metrics once; over/under-fitting & error discussion | **Model Lab**, **Test Evaluation** |
| 7 | Final pipeline saved with joblib | **Race Predictor** loads `models/final_pipeline.joblib` |
""")
st.info("Use the sidebar to move through the story: data → pipeline → model selection → test result → live predictions.")
