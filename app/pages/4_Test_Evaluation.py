"""Test Evaluation: the one-shot 2024-25 result, error analysis, slices, calibration."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_utils import (CLASS_COLORS, CLASS_ORDER, FIG, RED, RES, TARGET, card, config, header, load_csv, load_json,
                       load_test_predictions, require, setup_page, style_fig)
from f1ml.evaluation import error_distance, per_class_report, summary_metrics
from f1ml.modelling import GridRuleBaseline

setup_page("Test Evaluation")
header("Test Evaluation", "Final Random Forest scored once on the 2024–25 seasons (958 entries, 48 races)")
if not require([RES / "final_results.json", RES / "test_predictions.csv", RES / "test_confusion_matrix.csv"]):
    st.stop()

res = load_json(str(RES / "final_results.json"))
t = load_test_predictions()
rule_pred = GridRuleBaseline().fit(t).predict(t)
rule = summary_metrics(t[TARGET], rule_pred)
rf = summary_metrics(t[TARGET], t.predicted)
err = error_distance(t[TARGET], t.predicted)

c1, c2, c3, c4 = st.columns(4)
with c1:
    card("Accuracy", f"{rf['accuracy']:.1%}", f"grid rule {rule['accuracy']:.1%}")
with c2:
    card("Macro-F1", f"{rf['macro_f1']:.3f}", f"grid rule {rule['macro_f1']:.3f}")
with c3:
    card("Balanced accuracy", f"{rf['balanced_accuracy']:.3f}", f"grid rule {rule['balanced_accuracy']:.3f}")
with c4:
    card("ROC-AUC (OvR macro)", f"{res['test']['roc_auc_ovr_macro']:.3f}", "threshold-free quality")

st.write("")
st.info(f"**Reading this honestly.** Validation (2022–23) macro-F1 was {res['validation']['macro_f1']:.3f} and the test "
        f"score is {rf['macro_f1']:.3f} — no collapse, so the model did not overfit its selection data. But the "
        f"'finish where you start' rule scores {(rule['accuracy'] - rf['accuracy']) * 100:.1f} points higher in "
        f"accuracy on these two seasons. With one fixed split of 48 races there is no confidence interval, so the "
        f"size of this gap cannot be judged either way; it is reported rather than explained away. "
        f"{err['adjacent_share_of_errors']:.0%} of the "
        f"model's {err['errors']} errors are to the *neighbouring* category.")

tabs = st.tabs(["Confusion matrix", "Per class", "Where it works", "By race", "Calibration", "Feature importance",
                "Saved figures"])

with tabs[0]:
    cm = load_csv("test_confusion_matrix.csv", index_col=0)
    arr = cm.to_numpy()
    norm = st.toggle("Show row percentages (recall)", value=True)
    pct = arr / arr.sum(axis=1, keepdims=True)
    text = [[f"{arr[i, j]}<br>({pct[i, j]:.0%})" for j in range(3)] for i in range(3)]
    fig = go.Figure(go.Heatmap(z=pct if norm else arr, x=CLASS_ORDER, y=CLASS_ORDER, text=text,
                               texttemplate="%{text}", colorscale=[[0, "#1F1F2B"], [1, RED]], showscale=False))
    fig.update_layout(title="Confusion matrix — test 2024–25", xaxis_title="Predicted", yaxis_title="Actual",
                      yaxis_autorange="reversed")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.plotly_chart(style_fig(fig, 470), width="stretch")
    with c2:
        st.metric("Adjacent-class errors", f"{err['adjacent_class_errors']} of {err['errors']}")
        st.metric("Top 3 ↔ Outside Points errors", err["top3_vs_outside_errors"])
        st.markdown("The model rarely makes a *big* mistake: it almost never confuses a podium with no points. "
                    "Most errors sit on the **Points** boundary (P10 vs P11, P3 vs P4) — exactly where the "
                    "category definition is most arbitrary.")

with tabs[1]:
    pc = per_class_report(t[TARGET], t.predicted)
    rule_pc = per_class_report(t[TARGET], rule_pred)
    long = pd.concat([pc.assign(model="Random Forest"), rule_pc.assign(model="Grid rule")]).reset_index(names="class")
    long = long.melt(id_vars=["class", "model"], value_vars=["precision", "recall", "f1"], var_name="metric")
    fig = px.bar(long, x="class", y="value", color="model", facet_col="metric", barmode="group", text_auto=".2f",
                 category_orders={"class": CLASS_ORDER}, color_discrete_sequence=[RED, "#9AA0A6"])
    fig.update_layout(yaxis=dict(range=[0, 1]), xaxis_title=None)
    fig.for_each_xaxis(lambda a: a.update(title=None))
    st.plotly_chart(style_fig(fig, 420), width="stretch")
    st.dataframe(pc.round(3), width="stretch")
    st.caption("Points (P4–P10) is the hardest class — a wide, crowded middle — while Outside Points is easiest.")

with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        gb = load_csv("test_by_grid_bin.csv")
        fig = px.bar(gb, x="grid_bin", y="accuracy", text_auto=".0%", color_discrete_sequence=[RED],
                     hover_data=["n"], title="Accuracy by starting-grid bin")
        fig.update_layout(yaxis=dict(range=[0, 1], tickformat=".0%"), xaxis_title="Grid position")
        st.plotly_chart(style_fig(fig, 380), width="stretch")
        st.caption("Weakest in the midfield (P4–P10) where strategy and incidents reshuffle results; strongest at the back.")
    with c2:
        bs = load_csv("test_by_season.csv")
        long = bs.melt(id_vars="season", value_vars=["accuracy", "macro_f1", "balanced_accuracy"])
        fig = px.bar(long, x="season", y="value", color="variable", barmode="group", text_auto=".3f",
                     color_discrete_sequence=["#7A7F8C", RED, "#3D85D1"], title="Test metrics by season")
        fig.update_layout(yaxis=dict(range=[0, 1]), xaxis=dict(type="category"))
        st.plotly_chart(style_fig(fig, 380), width="stretch")
    st.markdown("**Classified finishers vs retirements**")
    fs = load_csv("test_by_finish_status.csv")
    fs = fs.rename(columns={"nc": "group", "size": "entries", "mean": "accuracy"}).assign(accuracy=lambda d: d.accuracy * 100)
    st.dataframe(fs, width="stretch", hide_index=True,
                 column_config={"accuracy": st.column_config.NumberColumn("Accuracy", format="%.1f%%")})
    st.caption("Accuracy drops for DNF/DSQ/DNS entries — retirements cannot be foreseen from pre-race data.")

with tabs[3]:
    per = (t.assign(rule_correct=(rule_pred == t[TARGET]).to_numpy())
           .groupby(["season", "round", "race_name"]).agg(RF=("correct", "mean"), Rule=("rule_correct", "mean"),
                                                         entries=("correct", "size")).reset_index())
    per["label"] = per.season.astype(str) + " R" + per["round"].astype(int).astype(str).str.zfill(2) + " " + per.race_name
    long = per.melt(id_vars=["label", "season", "round"], value_vars=["RF", "Rule"], var_name="model", value_name="accuracy")
    fig = px.line(long, x="label", y="accuracy", color="model", markers=True,
                  color_discrete_map={"RF": RED, "Rule": "#9AA0A6"}, title="Accuracy per race: Random Forest vs grid rule")
    fig.update_layout(xaxis=dict(showticklabels=False, title="48 test races in calendar order"),
                      yaxis=dict(range=[0, 1.02], tickformat=".0%"))
    st.plotly_chart(style_fig(fig, 420), width="stretch")
    wins = int((per.RF > per.Rule).sum()); ties = int((per.RF == per.Rule).sum()); loss = int((per.RF < per.Rule).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Races RF better", wins); c2.metric("Tied", ties); c3.metric("Races grid rule better", loss)
    st.dataframe(per.drop(columns="label").round(3), width="stretch", hide_index=True)

with tabs[4]:
    st.markdown("Is a predicted 70 % really right about 70 % of the time? Reliability of the model's probabilities "
                "on the test set (10 equal-width bins).")
    cls_sel = st.radio("Class", CLASS_ORDER, horizontal=True)
    p = t[f"proba_{cls_sel}"].to_numpy(); y = (t[TARGET] == cls_sel).to_numpy()
    bins = np.clip((p * 10).astype(int), 0, 9)
    cal = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin").agg(mean_pred=("p", "mean"),
                                                                       observed=("y", "mean"), n=("y", "size")).reset_index()
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#888"), name="perfect")
    fig.add_scatter(x=cal.mean_pred, y=cal.observed, mode="lines+markers", marker=dict(size=cal.n.clip(upper=200) / 8 + 6),
                    line_color=CLASS_COLORS[cls_sel], name=cls_sel, text=cal.n, hovertemplate="pred %{x:.2f}<br>obs %{y:.2f}<br>n=%{text}")
    fig.update_layout(title=f"Reliability — {cls_sel}", xaxis=dict(title="Predicted probability", range=[0, 1]),
                      yaxis=dict(title="Observed frequency", range=[0, 1]))
    st.plotly_chart(style_fig(fig, 460), width="stretch")
    st.caption("Marker size = number of entries in the bin. Thin bins at the extremes are noisy.")

with tabs[5]:
    imp = load_csv("feature_importance.csv", index_col=0)
    top = imp[~imp.index.str.startswith("missingindicator_")].head(15).iloc[::-1].reset_index()
    top.columns = ["feature", "importance"]
    fig = px.bar(top, x="importance", y="feature", orientation="h", color_discrete_sequence=[RED],
                 title="Top 15 features — Random Forest impurity importance")
    fig.update_layout(yaxis_title=None)
    st.plotly_chart(style_fig(fig, 520), width="stretch")
    st.caption("Grid and qualifying position dominate, followed by team form (`constructor_points_last5`) and the gap "
               "to pole. Correlated inputs split importance between them.")

with tabs[6]:
    figs = {"test_confusion_matrix.png": "Confusion matrix", "test_accuracy_by_grid.png": "Accuracy by grid",
            "model_comparison.png": "Model comparison", "rf_depth_curve.png": "RF depth curve",
            "rf_feature_importance.png": "RF feature importance"}
    pick = st.selectbox("Figure", [k for k in figs if (FIG / k).exists()], format_func=figs.get)
    if pick:
        st.image(str(FIG / pick), width="stretch")
