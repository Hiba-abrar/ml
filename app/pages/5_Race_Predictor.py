"""Race Predictor: run the saved final pipeline on any race, with what-if edits of grid / qualifying."""
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import app_utils as U
from app_utils import CLASS_COLORS, CLASS_ORDER, FEATURE_INFO, RED, TARGET, config, header, pill, setup_page, style_fig

setup_page("Race Predictor")
header("Race Predictor", "The saved final pipeline (features → imputer → scaler → Random Forest) on any race, pre-race inputs only")
if not U.require([config.CLEAN_FILE, config.MODELS_DIR / "final_pipeline.joblib"]):
    st.stop()

df = U.load_dataset()
pipe = U.load_pipeline()
mc = U.load_json(str(config.MODELS_DIR / "model_card.json"))
st.caption(f"Random Forest · fitted on seasons {mc['fitted_on_seasons']} · test accuracy "
           f"{mc['test']['accuracy']:.1%} · test macro-F1 {mc['test']['macro_f1']:.3f}")


@st.cache_data(show_spinner=False)
def run(race_id: str, overrides_json: str):
    return U.predict_race(race_id, json.loads(overrides_json) or None)


c1, c2 = st.columns([1, 2])
seasons = sorted(df.season.unique(), reverse=True)
season = c1.selectbox("Season", seasons)
races = df[df.season == season].drop_duplicates("race_id").sort_values("round")
labels = {r.race_id: f"R{int(r.round):02d} · {r.race_name}" for r in races.itertuples()}
race_id = c2.selectbox("Race", list(labels), format_func=labels.get)

if season in config.TEST_SEASONS:
    st.success("🟢 **True out-of-sample.** This race belongs to the held-out test seasons (2024–25): the model never saw it.")
elif season in config.VALID_SEASONS:
    st.info("🟡 **Validation season (2022–23).** Not used to fit the model's parameters, but used to compare and select models.")
else:
    st.warning("🟠 **Training season (2010–21).** The model was fitted on this race, so predictions here are in-sample "
               "and look better than they would on new races.")

base, _ = run(race_id, "{}")
race = df[df.race_id == race_id].sort_values("grid").reset_index(drop=True)

# ---------------------------------------------------------------- what-if grid editor
with st.expander("🔧 What-if: change grid / qualifying positions", expanded=False):
    st.caption("Edit starting grid or qualifying position for any driver. `grid_penalty_places` is recomputed and "
               "the pipeline predicts the whole race again. Pre-race features such as recent form stay as they were.")
    ed = race[["driver_id", "driver_name", "constructor_name", "grid", "quali_position"]].copy()
    edited = st.data_editor(ed, hide_index=True, width="stretch", key=f"ed_{race_id}",
                            disabled=["driver_id", "driver_name", "constructor_name"],
                            column_config={"driver_id": None,
                                           "driver_name": "Driver", "constructor_name": "Team",
                                           "grid": st.column_config.NumberColumn("Grid", min_value=0, max_value=26, step=1),
                                           "quali_position": st.column_config.NumberColumn("Quali", min_value=1, max_value=26, step=1)})
    overrides = {}
    for o, n in zip(ed.itertuples(), edited.itertuples()):
        d = {}
        if n.grid != o.grid:
            d["grid"] = float(n.grid)
        if not (pd.isna(n.quali_position) and pd.isna(o.quali_position)) and n.quali_position != o.quali_position:
            d["quali_position"] = float(n.quali_position)
        if d:
            overrides[o.driver_id] = d
    if overrides:
        st.success(f"{len(overrides)} driver(s) changed — showing the scenario below.")

scen, feats = run(race_id, json.dumps(overrides, sort_keys=True))
scen = scen.sort_values("grid").reset_index(drop=True)

# ---------------------------------------------------------------- race summary
acc = (scen.predicted == scen[TARGET]).mean()
act_podium = set(scen[scen.finish_position.between(1, 3)].driver_name)
pred_podium = scen.nlargest(3, "proba_Top 3").driver_name.tolist()
hits = len(act_podium & set(pred_podium))
m1, m2 = st.columns(2)
m1.metric("Category accuracy (this race)", f"{acc:.0%}" if not overrides else f"{acc:.0%} (what-if)")
m2.metric("Podium picks correct", f"{hits} / 3", help="Top 3 drivers by P(Top 3) vs the actual podium")
actual_names = scen[scen.finish_position.between(1, 3)].sort_values("finish_position").driver_name.tolist()
st.markdown(f"**Model's most likely podium:** {', '.join(pred_podium)}  \n"
            f"**Actual podium:** {', '.join(actual_names) if actual_names else 'n/a'}")

tab_grid, tab_driver = st.tabs(["Whole grid", "Driver deep-dive"])

with tab_grid:
    long = scen.melt(id_vars=["driver_name", "grid"], value_vars=[f"proba_{c}" for c in CLASS_ORDER],
                     var_name="class", value_name="p")
    long["class"] = long["class"].str.replace("proba_", "", regex=False)
    order = scen.driver_name.tolist()[::-1]          # plotly draws the first category at the bottom
    fig = px.bar(long, y="driver_name", x="p", color="class", orientation="h", color_discrete_map=CLASS_COLORS,
                 category_orders={"class": CLASS_ORDER, "driver_name": order},
                 title="Predicted probabilities, in grid order (pole at top)")
    fig.update_layout(xaxis=dict(tickformat=".0%", title=None))
    style_fig(fig, max(480, 26 * len(scen)))
    # explicit order, applied last: plotly draws the first category at the bottom, so pole ends up on top
    fig.update_yaxes(title=None, categoryorder="array", categoryarray=order)
    st.plotly_chart(fig, width="stretch")
    show = scen.assign(**{"✓": np.where(scen.predicted == scen[TARGET], "✓", "✗"),
                          "Actual result": scen.apply(lambda r: f"P{int(r.finish_position)}" if pd.notna(r.finish_position)
                                                      else r.finish_status, axis=1)})
    show = show[["grid", "driver_name", "constructor_name", "predicted", "proba_Top 3", "proba_Points",
                 "proba_Outside Points", "Actual result", TARGET, "✓"]]
    show = show.assign(**{f"proba_{c}": show[f"proba_{c}"] * 100 for c in CLASS_ORDER})   # shown as percentages
    st.dataframe(show, width="stretch", hide_index=True, column_config={
        "grid": "Grid", "driver_name": "Driver", "constructor_name": "Team", "predicted": "Predicted",
        TARGET: "Actual category",
        **{f"proba_{c}": st.column_config.ProgressColumn(f"P({c})", format="%.0f%%", min_value=0, max_value=100)
           for c in CLASS_ORDER}})

with tab_driver:
    drv_labels = dict(zip(scen.driver_id, scen.driver_name + " (grid " + scen.grid.astype(int).astype(str) + ")"))
    drv = st.selectbox("Driver", list(drv_labels), format_func=drv_labels.get)
    r = scen[scen.driver_id == drv].iloc[0]
    b = base[base.driver_id == drv].iloc[0]
    left, right = st.columns([1, 1.5])
    with left:
        st.markdown(f"**Predicted:** {pill(r.predicted)}", unsafe_allow_html=True)
        for c in CLASS_ORDER:
            delta = r[f"proba_{c}"] - b[f"proba_{c}"]
            st.metric(f"P({c})", f"{r[f'proba_{c}']:.1%}", delta=f"{delta:+.1%}" if overrides and abs(delta) > 1e-9 else None)
        res = f"P{int(r.finish_position)}" if pd.notna(r.finish_position) else r.finish_status
        st.markdown(f"**Actual:** {res} → {pill(r[TARGET])}", unsafe_allow_html=True)
        st.caption(f"Probability the model gave to what actually happened: **{r['proba_' + r[TARGET]]:.1%}**")
    with right:
        probs = pd.DataFrame({"category": CLASS_ORDER, "probability": [r[f"proba_{c}"] for c in CLASS_ORDER]})
        fig = px.bar(probs, x="probability", y="category", orientation="h", color="category", text="probability",
                     color_discrete_map=CLASS_COLORS, category_orders={"category": CLASS_ORDER}, title="Class probabilities")
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig.update_layout(showlegend=False, xaxis=dict(range=[0, 1.1], tickformat=".0%", title=None), yaxis_title=None)
        st.plotly_chart(style_fig(fig, 280), width="stretch")
    st.subheader("The 20 inputs the model saw")
    fr = feats[feats.driver_id == drv].iloc[0]
    # field context: where this driver's value sits among the race's entrants
    ctx = feats[config.FEATURES]
    table = pd.DataFrame({"feature": config.FEATURES,
                          "value": [fr[f] for f in config.FEATURES],
                          "field median": [ctx[f].median() for f in config.FEATURES],
                          "description": [FEATURE_INFO[f][0] for f in config.FEATURES]})
    st.dataframe(table, width="stretch", hide_index=True, column_config={
        "value": st.column_config.NumberColumn(format="%.2f"), "field median": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Missing values (e.g. first visit to a circuit) are filled by the pipeline's median imputer and flagged with "
               "an indicator. All values come from `PreRaceFeatureBuilder` — strictly earlier races only.")

st.caption("Probabilities express the model's uncertainty given historical patterns, not guarantees: crashes, safety "
           "cars and strategy calls are not visible before the start.")
