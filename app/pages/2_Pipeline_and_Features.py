"""Pipeline & Features: the leakage-safe feature builder, the 20 inputs, tests and provenance checks."""
import ast
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app_utils import (FEATURE_INFO, PROJECT_ROOT, RED, RES, TARGET, card, config, header, load_csv, load_json,
                       load_pipeline, load_test_predictions, require, setup_page, style_fig)

setup_page("Pipeline & Features")
header("Pipeline & Features", "One scikit-learn Pipeline: feature engineering → imputation/scaling → classifier")
if not require([config.MODELS_DIR / "final_pipeline.joblib", config.MODELS_DIR / "model_card.json"]):
    st.stop()

pipe = load_pipeline()
mc = load_json(str(config.MODELS_DIR / "model_card.json"))
prep = pipe.named_steps["prep"]
n_transformed = len(prep.get_feature_names_out())

c1, c2, c3, c4 = st.columns(4)
with c1:
    card("Raw features", str(mc["n_input_features"]), "20 engineered pre-race inputs")
with c2:
    card("After imputation", str(n_transformed), "incl. missing-value indicators")
with c3:
    card("Fitted on seasons", mc["fitted_on_seasons"], "medians, scaler and trees")
with c4:
    card("History table", f"through {mc['history_table_through_season']}", "look-up of past results only")

st.write("")
tab_pipe, tab_feat, tab_leak, tab_prov = st.tabs(
    ["Pipeline", "The 20 features", "Leakage safeguards & tests", "Provenance check"])

with tab_pipe:
    st.markdown("```text\n" + "\n".join(f"{i + 1}. {n:<9}→ {type(s).__name__}" for i, (n, s) in
                                         enumerate(pipe.steps)) + "\n```")
    st.markdown(
        """
* **`PreRaceFeatureBuilder`** (`f1ml.features`) — receives the *raw* race-entry table and returns the 20 model
  inputs. It learns **no parameters**: `fit` just stores a history table of past results, and rolling statistics
  look only at strictly earlier races (`shift(1)` inside driver, driver+circuit and team groups; standings after the
  *previous* round).
* **`ColumnTransformer`** — median `SimpleImputer(add_indicator=True)` then `StandardScaler`, fitted on training
  rows only (inside CV: on each training fold only).
* **`RandomForestClassifier`** — the final model, saved together with the two steps above, so the app and
  `predict_example.py` feed it *raw* race entries and need no external encoders.
""")
    st.markdown("**Final model hyper-parameters (from `model_card.json`)**")
    st.json(mc["hyperparameters"], expanded=False)
    st.caption(f"Probability column order of the saved model: {list(pipe.classes_)} (alphabetical, as scikit-learn "
               f"fits them); report order everywhere else is {config.CLASS_ORDER}.")

with tab_feat:
    corr = load_csv("eda_correlation_with_target.csv", index_col=0)
    imp = load_csv("feature_importance.csv", index_col=0)
    rows = []
    for f in config.FEATURES:
        desc, fam = FEATURE_INFO[f]
        rows.append({"feature": f, "family": fam, "description": desc,
                     "Spearman ρ vs target": float(corr.loc[f, "spearman_with_target"]) if corr is not None else np.nan,
                     "RF importance": float(imp.loc[f, "importance"]) if imp is not None and f in imp.index else np.nan})
    ft = pd.DataFrame(rows).sort_values("RF importance", ascending=False)
    c1, c2 = st.columns([1.3, 1])
    with c1:
        top = ft.sort_values("RF importance")
        fig = px.bar(top, x="RF importance", y="feature", orientation="h", color="family",
                     title="Random Forest importance (final model)")
        fig.update_layout(yaxis_title=None, xaxis_title="Mean decrease in impurity")
        st.plotly_chart(style_fig(fig, 620), width="stretch")
    with c2:
        fam = ft.groupby("family")["RF importance"].sum().reset_index()
        fig = px.pie(fam, names="family", values="RF importance", hole=.5, title="Importance by feature family")
        st.plotly_chart(style_fig(fig, 340), width="stretch")
        st.caption("Qualifying & grid features carry over half of the total importance; recent form and championship "
                   "position add most of the rest. Correlated features (grid vs quali position) share importance.")
        if imp is not None:
            mi = imp[imp.index.str.startswith("missingindicator_")].importance.sum()
            st.metric("Total importance of missing-value indicators", f"{mi:.2%}")
    st.dataframe(ft, width="stretch", hide_index=True, column_config={
        "Spearman ρ vs target": st.column_config.NumberColumn(format="%.2f"),
        "RF importance": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=float(ft["RF importance"].max()))})
    st.caption("`teammate_prev_finish` — the other car's result in the previous race — is the extra feature added "
               "in the final version.")

with tab_leak:
    st.markdown(
        """
| Risk | Safeguard |
|---|---|
| Random split mixes future and past | Season-based chronological split: 2010–21 / 2022–23 / 2024–25 (asserted in `data.split_data`) |
| CV folds peek ahead | Expanding-window `TimeSeriesSplit` over **race weekends** — all drivers of a race stay together, validation folds are always later |
| Preprocessing learns from test data | Imputer, scaler and forest are *pipeline steps* → fitted on training rows only |
| Rolling features include the race itself | Every statistic is a `shift(1)` look-back; teammate feature uses the teammate's **previous** race |
| Test set tuned against | Test seasons are loaded but not touched until one final cell; metrics reported once |
""")
    tests = []
    for p in sorted((PROJECT_ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in tree.body:
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"):
                tests.append({"file": p.name, "test": n.name.replace("test_", "").replace("_", " ")})
    st.subheader(f"Automated checks ({len(tests)} tests)")
    st.dataframe(pd.DataFrame(tests), width="stretch", hide_index=True)
    if st.button("▶ Run the test suite now"):
        with st.spinner("Running pytest (this can take a minute)…"):
            try:
                r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=PROJECT_ROOT,
                                   capture_output=True, text=True, timeout=600)
                (st.success if r.returncode == 0 else st.error)("All tests passed" if r.returncode == 0
                                                                else "Some tests failed")
                st.code((r.stdout or "") + (r.stderr or ""), language=None)
            except Exception as exc:  # pragma: no cover
                st.error(f"Could not run pytest: {exc}")

with tab_prov:
    st.markdown("`provenance/frozen_reference/` holds the earlier frozen model and its saved predictions. "
                "The rebuilt pipeline is meant to reproduce them **exactly**. Checked live from the saved files:")
    fz_path = config.FROZEN_DIR / "test_predictions.csv"
    if fz_path.exists():
        fz = pd.read_csv(fz_path)
        new = load_test_predictions()
        m = fz.merge(new, on=["race_id", "driver_id"], suffixes=("_ref", "_new"))
        same = (m["predicted_class"] == m["predicted"]).sum()
        pdiff = max(np.abs(m[f"probability_{c}"] - m[f"proba_{c}"]).max() for c in config.CLASS_ORDER)
        c1, c2, c3 = st.columns(3)
        c1.metric("Test predictions identical", f"{same} / {len(m)}")
        c2.metric("Max probability difference", f"{pdiff:.1e}")
        c3.metric("Rows compared", len(m))
    for name, label in [("configuration.json", "Frozen model configuration"),
                        ("test_metrics.json", "Frozen reference test metrics")]:
        d = load_json(str(config.FROZEN_DIR / name))
        if d is not None:
            with st.expander(label):
                st.json(d, expanded=False)
    src = load_json(str(config.DATA_DIR / "data_sources.json"))
    if src:
        with st.expander("Data sources & licences"):
            st.json(src, expanded=False)
