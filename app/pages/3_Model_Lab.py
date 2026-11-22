"""Model Lab: baselines vs algorithms, 5-fold CV grid search, over/under-fitting, final RF folds."""
import ast
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_utils import MODEL_COLORS, RED, RES, card, config, header, load_csv, load_json, require, setup_page, style_fig

setup_page("Model Lab")
header("Model Lab", "Three baselines and four tuned algorithms, compared on train, 5-fold CV and validation")
need = [RES / f for f in ["model_comparison.csv", "model_comparison_table.csv", "best_params_gridsearch.json",
                          "final_rf_cv_folds.csv", "final_results.json"]]
if not require(need):
    st.stop()

cmp_long = load_csv("model_comparison.csv")
table = load_csv("model_comparison_table.csv")
best = json.loads((RES / "best_params_gridsearch.json").read_text())
folds = load_csv("final_rf_cv_folds.csv")
final = load_json(str(RES / "final_results.json"))
sel = final["rf_grid_position_of_final_spec"]

top = table.sort_values("macro-F1 validation", ascending=False).iloc[0]
knn = table[table.model == "KNN"].iloc[0]
c1, c2, c3, c4 = st.columns(4)
with c1:
    card("Best on validation", top["model"], f"macro-F1 {top['macro-F1 validation']:.3f}")
with c2:
    card("Best grid-only baseline", f"{table[table.model == 'Baseline: grid rule']['macro-F1 validation'].iloc[0]:.3f}",
         "grid rule, validation macro-F1")
with c3:
    card("Most overfit", "KNN", f"train {knn['macro-F1 train']:.2f} → val {knn['macro-F1 validation']:.2f}")
with c4:
    card("Final RF CV rank", f"{sel['final_spec_cv_rank']} / {sel['n_configs']}",
         f"{sel['final_spec_cv_macro_f1']:.3f} vs best {sel['best_cv_macro_f1']:.3f}")

st.write("")
tab_cmp, tab_fit, tab_cv, tab_final = st.tabs(
    ["Comparison", "Over / under-fitting", "Hyper-parameter search", "Final model folds"])

with tab_cmp:
    long = table.melt(id_vars="model", value_vars=["macro-F1 train", "macro-F1 CV", "macro-F1 validation"],
                      var_name="stage", value_name="macro-F1").dropna()
    fig = px.bar(long, x="model", y="macro-F1", color="stage", barmode="group", text_auto=".3f",
                 category_orders={"model": table.model.tolist()},
                 color_discrete_sequence=["#7A7F8C", "#3D85D1", RED], title="Macro-F1 by stage")
    fig.update_layout(yaxis=dict(range=[0, 1.05]), xaxis_title=None)
    st.plotly_chart(style_fig(fig, 460), width="stretch")
    st.dataframe(table.set_index("model").round(3), width="stretch")
    st.caption("Accuracy alone is misleading ('Outside Points' is ~half of all rows); macro-F1 weights the three "
               "classes equally. Every model beats the majority baseline, and on validation the full-feature "
               "models lead the grid rule by up to ~6 macro-F1 points; on the test seasons the rule came out ahead "
               "(see Test Evaluation).")
    metric = st.radio("Detail metric", ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"], horizontal=True)
    sp = cmp_long[cmp_long.split.isin(["train", "validation"])]
    fig = px.bar(sp, x="model", y=metric, color="split", barmode="group", text_auto=".3f",
                 color_discrete_sequence=["#7A7F8C", RED], category_orders={"model": table.model.tolist()})
    fig.update_layout(yaxis=dict(range=[0, 1.05]), xaxis_title=None)
    st.plotly_chart(style_fig(fig, 400), width="stretch")

with tab_fit:
    gap = table.dropna(subset=["macro-F1 train"]).sort_values("train - validation gap")
    fig = go.Figure()
    fig.add_bar(y=gap.model, x=gap["macro-F1 train"], name="Train", orientation="h", marker_color="#7A7F8C")
    fig.add_bar(y=gap.model, x=gap["macro-F1 validation"], name="Validation", orientation="h", marker_color=RED)
    fig.update_layout(barmode="group", title="Train vs validation macro-F1", xaxis=dict(range=[0, 1.05]))
    st.plotly_chart(style_fig(fig, 460), width="stretch")
    st.markdown(
        f"""
* **Overfitting:** KNN reaches a perfect training score (**{knn['macro-F1 train']:.2f}**) because each training point
  is its own nearest neighbour under distance weighting, yet drops to {knn['macro-F1 validation']:.2f} on
  validation (gap **{knn['train - validation gap']:.2f}**). The tuned RF, SVM and Logistic Regression all keep a gap of ≈ 0.01–0.02.
* **Underfitting / low ceiling:** the majority baseline is low on both train and validation; the grid-only models
  sit just below the full-feature models, showing how much of the task the grid explains alone.
* Shallow, leaf-constrained forests (depth 6, 40 samples per leaf) were chosen to keep variance low.
""")

with tab_cv:
    files = {"Logistic Regression": "cv_results_logistic_regression.csv", "KNN": "cv_results_knn.csv",
             "SVM (RBF)": "cv_results_svm_rbf.csv", "Random Forest": "cv_results_random_forest.csv"}
    model = st.selectbox("Algorithm", list(files), index=3)
    cv = load_csv(files[model])
    pcols = [c for c in cv.columns if c.startswith("param_")]
    cv["config"] = cv["params"].map(lambda s: ", ".join(f"{k.replace('clf__', '')}={v}"
                                                        for k, v in ast.literal_eval(s).items()))
    bp = ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in best[model].items())
    st.markdown(f"**Best parameters (GridSearchCV, 5-fold expanding window, macro-F1):** `{bp}`")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(cv, x="mean_train_score", y="mean_test_score", color="rank_test_score",
                         color_continuous_scale="Reds_r", hover_name="config", error_y="std_test_score",
                         title="Every configuration: train vs CV macro-F1")
        lim = [min(cv.mean_test_score.min(), cv.mean_train_score.min()) - .02, 1.0]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color="#888"),
                                 showlegend=False, hoverinfo="skip"))
        fig.update_layout(xaxis_title="Train macro-F1", yaxis_title="CV macro-F1")
        st.plotly_chart(style_fig(fig, 420), width="stretch")
        st.caption("Points far below the dashed line overfit.")
    with c2:
        if model == "Random Forest":
            d = (cv.groupby("param_clf__max_depth")[["mean_train_score", "mean_test_score"]].max().reset_index())
            fig = go.Figure()
            fig.add_scatter(x=d.param_clf__max_depth, y=d.mean_train_score, name="Train", mode="lines+markers",
                            line_color="#7A7F8C")
            fig.add_scatter(x=d.param_clf__max_depth, y=d.mean_test_score, name="CV", mode="lines+markers",
                            line_color=RED)
            fig.update_layout(title="Best score per max_depth", xaxis=dict(title="max_depth", dtick=1),
                              yaxis_title="macro-F1")
            st.plotly_chart(style_fig(fig, 420), width="stretch")
            st.caption("Training score keeps climbing with depth while CV score flattens — the classic overfitting curve.")
        else:
            fig = px.bar(cv.sort_values("rank_test_score").head(10).iloc[::-1], x="mean_test_score", y="config",
                         orientation="h", error_x="std_test_score", color_discrete_sequence=[RED],
                         title="Top 10 configurations (CV macro-F1)")
            fig.update_layout(yaxis_title=None, xaxis=dict(range=[0, 1]))
            st.plotly_chart(style_fig(fig, 420), width="stretch")
    show = cv.sort_values("rank_test_score")[["rank_test_score", "config", "mean_test_score", "std_test_score",
                                              "mean_train_score", "mean_fit_time"]].head(15)
    st.dataframe(show.round(4), width="stretch", hide_index=True)

with tab_final:
    st.markdown(f"The final Random Forest (400 trees, depth 6, leaf 40, max_features 0.3) is kept as specified. In the "
                f"72-configuration grid it ranks **{sel['final_spec_cv_rank']}th** "
                f"({sel['final_spec_cv_macro_f1']:.3f} vs {sel['best_cv_macro_f1']:.3f} for the best) — within CV noise.")
    long = folds.melt(id_vars=["fold", "train_rows"], value_vars=["train_macro_f1", "cv_macro_f1"],
                      var_name="score", value_name="macro-F1")
    long["score"] = long.score.map({"train_macro_f1": "Train fold", "cv_macro_f1": "Validation fold"})
    fig = px.line(long, x="fold", y="macro-F1", color="score", markers=True, hover_data=["train_rows"],
                  color_discrete_sequence=["#7A7F8C", RED], title="Expanding-window 5-fold CV of the final model")
    fig.update_layout(xaxis=dict(dtick=1), yaxis=dict(range=[0.5, 0.8]))
    st.plotly_chart(style_fig(fig, 380), width="stretch")
    c1, c2 = st.columns(2)
    c1.metric("CV macro-F1 (mean ± std)", f"{final['final_rf_cv_macro_f1_mean']:.3f} ± {final['final_rf_cv_macro_f1_std']:.3f}")
    c2.metric("Validation macro-F1", f"{final['validation']['macro_f1']:.3f}")
    ocf = load_json(str(config.FROZEN_DIR / "original_cv_folds.json"))
    if ocf:
        o = pd.DataFrame(ocf)
        o["train_end"] = o.train_end.str[:10]; o["eval_start"] = o.eval_start.str[:10]; o["eval_end"] = o.eval_end.str[:10]
        st.markdown("**Fold boundaries (each validation block lies strictly after its training block):**")
        st.dataframe(o, width="stretch", hide_index=True)
