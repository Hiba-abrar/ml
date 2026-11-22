"""Surprise Analysis: which results did the model find unlikely, and which races were chaos?"""
import pandas as pd
import plotly.express as px
import streamlit as st

import app_utils as U
from app_utils import CLASS_COLORS, CLASS_ORDER, RED, TARGET, config, header, setup_page, style_fig

setup_page("Surprise Analysis")
header("Surprise Analysis", "Which results did the model consider unlikely — and which races were chaos?")
if not U.require([U.RES / "test_predictions.csv"]):
    st.stop()

test = U.load_test_predictions().assign(dataset="Test 2024–25 · final model")
val = U.load_validation_predictions()
with st.sidebar:
    st.header("Analysis set")
    options = ["Test 2024–25"] + (["+ Validation 2022–23"] if val is not None else [])
    chosen = st.radio("Seasons", options, help="Validation predictions come from the frozen reference Random Forest "
                      "(fitted 2010–21), which the final pipeline reproduces exactly. Every prediction is out-of-sample.")
frame = test if chosen == "Test 2024–25" else pd.concat([val.assign(dataset="Validation 2022–23 · reference RF"), test],
                                                        ignore_index=True)
with st.sidebar:
    st.header("Filters")
    seasons = sorted(frame.season.unique())
    sel_seasons = st.multiselect("Season", seasons, default=seasons)
    circuits = sorted(frame.circuit_name.dropna().unique())
    sel_circuits = st.multiselect("Circuit", circuits, default=[], placeholder="All circuits")
    top_n = st.slider("Rows / bars to show", 5, 50, 15)
    thr = st.slider("Surprise threshold: P(actual) below", 0.05, 0.30, 0.15, 0.01)

d = frame[frame.season.isin(sel_seasons)]
if sel_circuits:
    d = d[d.circuit_name.isin(sel_circuits)]
if d.empty:
    st.warning("No results for this filter.")
    st.stop()
d = d.assign(is_surprise=d.p_actual < thr)

races = (d.groupby(["season", "round", "race_id", "race_name", "circuit_name"])
         .apply(lambda g: pd.Series({
             "mean_surprise": g.surprise.mean(), "max_surprise": g.surprise.max(),
             "n_surprises": int(g.is_surprise.sum()), "dnfs": int((g.finish_status != "Classified").sum()),
             "model_accuracy": g.correct.mean(),
             "biggest_shock_driver": g.loc[g.surprise.idxmax(), "driver_name"],
             "biggest_shock_result": g.loc[g.surprise.idxmax(), "actual_result"],
             "biggest_shock_grid": int(g.loc[g.surprise.idxmax(), "grid"])}), include_groups=False)
         .reset_index())

with st.expander("How the surprise score works"):
    st.markdown(r"""
For each driver in each race the model outputs three probabilities. We look up the probability of the category that
**actually happened**, $p_{actual}$, and define **surprise** $= -\ln(p_{actual})$ (p = 0.5 → 0.69, p = 0.1 → 2.3, p = 0.01 → 4.6).

* A single shock (pole-sitter retires) gives one very surprising **driver result**.
* A race is **unpredictable** when the *mean* surprise over all its drivers is high — many results differ from
  expectation at once (rain, safety cars, first-lap chaos).

**Limits.** Probabilities are the model's estimate, not certainty; they ignore information the model never sees (car
upgrades, team orders, strategy, late penalties). A "surprising" result can reveal a model weakness rather than a
genuinely shocking race, and ~2 races per circuit are a small sample.
""")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Driver results", f"{len(d):,}")
c2.metric("Races", races.race_id.nunique())
c3.metric(f"Surprises (p < {thr:.0%})", int(d.is_surprise.sum()))
c4.metric("Mean race surprise", f"{races.mean_surprise.mean():.2f}")

tab_drv, tab_race, tab_circ, tab_season = st.tabs(
    ["Most surprising results", "Most unpredictable races", "Circuits", "Seasons"])

with tab_drv:
    view = d.nlargest(top_n, "surprise").assign(**{"prob %": lambda x: x.p_actual * 100})
    st.dataframe(view[["season", "race_name", "driver_name", "constructor_name", "grid", "actual_result", TARGET,
                       "predicted", "prob %", "surprise"]].rename(columns={
        "race_name": "race", "driver_name": "driver", "constructor_name": "team", TARGET: "actual category"}).round(2),
        width="stretch", hide_index=True)
    fig = px.scatter(d, x="grid", y="p_actual", color=TARGET, color_discrete_map=CLASS_COLORS,
                     category_orders={TARGET: CLASS_ORDER}, opacity=.7,
                     hover_data=["driver_name", "race_name", "season", "actual_result", "predicted"],
                     title="Every result: grid slot vs probability of the actual outcome")
    fig.add_hline(y=thr, line_dash="dash", line_color=RED, annotation_text=f"surprise threshold ({thr:.0%})")
    fig.update_layout(xaxis=dict(dtick=1, title="Grid position"), yaxis=dict(tickformat=".0%", title="P(actual category)"))
    st.plotly_chart(style_fig(fig, 480), width="stretch")
    st.caption("Points at the bottom are the shocks: a back-marker on the podium, or a front-row starter outside the points.")

with tab_race:
    top = races.nlargest(top_n, "mean_surprise").assign(label=lambda x: x.season.astype(str) + " " + x.race_name)
    fig = px.bar(top.iloc[::-1], x="mean_surprise", y="label", orientation="h", color="n_surprises",
                 color_continuous_scale=["#3D3D4A", RED],
                 hover_data=["max_surprise", "dnfs", "model_accuracy", "biggest_shock_driver", "biggest_shock_result",
                             "biggest_shock_grid"], title="Most unpredictable races (mean surprise per driver)")
    fig.add_vline(x=races.mean_surprise.mean(), line_dash="dash", line_color="#A0A0B0", annotation_text="average")
    fig.update_layout(yaxis_title=None, xaxis_title="Mean surprise (−ln p)", coloraxis_colorbar=dict(title="# surprises"))
    st.plotly_chart(style_fig(fig, max(420, 28 * len(top))), width="stretch")
    st.dataframe(top[["season", "round", "race_name", "mean_surprise", "n_surprises", "dnfs", "model_accuracy",
                      "biggest_shock_driver", "biggest_shock_result", "biggest_shock_grid"]].round(3),
                 width="stretch", hide_index=True)

with tab_circ:
    circ = (races.groupby("circuit_name").agg(races=("race_id", "count"), mean_surprise=("mean_surprise", "mean"),
                                              std=("mean_surprise", "std"), surprises_per_race=("n_surprises", "mean"))
            .reset_index().sort_values("mean_surprise", ascending=False))
    fig = px.bar(circ.head(top_n).iloc[::-1], x="mean_surprise", y="circuit_name", orientation="h", error_x="std",
                 text="races", color_discrete_sequence=["#3D85D1"],
                 title="Circuit unpredictability (mean race surprise; label = number of races)")
    fig.update_traces(texttemplate="n=%{text}", textposition="inside")
    fig.update_layout(yaxis_title=None, xaxis_title="Mean race surprise")
    st.plotly_chart(style_fig(fig, max(420, 26 * min(top_n, len(circ)))), width="stretch")
    st.caption("With only 1–4 races per circuit, one chaotic race decides the ranking — add the validation seasons for a more stable view.")

with tab_season:
    seas = (races.groupby("season").agg(races=("race_id", "count"), mean_surprise=("mean_surprise", "mean"),
                                        surprises_per_race=("n_surprises", "mean"), accuracy=("model_accuracy", "mean"))
            .reset_index())
    a, b = st.columns(2)
    with a:
        fig = px.bar(seas, x="season", y="mean_surprise", text_auto=".2f", color_discrete_sequence=[RED],
                     title="Mean race surprise by season")
        fig.update_layout(xaxis=dict(type="category"))
        st.plotly_chart(style_fig(fig, 380), width="stretch")
    with b:
        fig = px.box(races, x="season", y="mean_surprise", points="all", hover_name="race_name",
                     color_discrete_sequence=["#3D85D1"], title="Distribution of race surprise")
        fig.update_layout(xaxis=dict(type="category"))
        st.plotly_chart(style_fig(fig, 380), width="stretch")
    st.dataframe(seas.round(3), width="stretch", hide_index=True)
