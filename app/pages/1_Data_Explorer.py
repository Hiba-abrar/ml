"""Data & EDA: interactive exploration + the backend's saved EDA tables and figures."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_utils import (CLASS_COLORS, CLASS_ORDER, FEATURE_INFO, FIG, RED, TARGET, all_race_features, config,
                       header, load_csv, load_dataset, require, setup_page, style_fig)

setup_page("Data & EDA")
header("Data & EDA", "2010–2025 race results, qualifying and the engineered pre-race features")
if not require([config.CLEAN_FILE]):
    st.stop()

df = load_dataset()
with st.sidebar:
    st.header("Filters")
    lo, hi = int(df.season.min()), int(df.season.max())
    seasons = st.slider("Seasons", lo, hi, (lo, hi))
data = df[df.season.between(*seasons)]
st.caption(f"{len(data):,} driver-race records · {data.race_id.nunique()} races · "
           f"{data.driver_id.nunique()} drivers · {data.circuit_id.nunique()} circuits")


def share(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    return frame.groupby(by)[TARGET].value_counts(normalize=True).rename("share").reset_index()


tabs = st.tabs(["Grid vs finish", "Class balance & seasons", "Feature signal", "Data quality",
                "Drivers & teams", "Circuits", "Weather", "Notebook figures"])

with tabs[0]:
    g = data[data.grid.between(1, 24)]
    fig = px.line(share(g, "grid"), x="grid", y="share", color=TARGET, markers=True,
                  category_orders={TARGET: CLASS_ORDER}, color_discrete_map=CLASS_COLORS,
                  title="Probability of each category from each grid slot")
    fig.update_layout(xaxis=dict(dtick=1, title="Grid position"), yaxis=dict(title="Share", tickformat=".0%"))
    st.plotly_chart(style_fig(fig), width="stretch")
    cls = g[g.finish_position.notna()]
    heat = pd.crosstab(cls.finish_position, cls.grid)
    fig = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="Reds",
                               hovertemplate="Grid %{x} → P%{y}<br>%{z} results<extra></extra>"))
    fig.add_trace(go.Scatter(x=[1, 24], y=[1, 24], mode="lines", line=dict(color="#F2F2F5", dash="dash", width=1),
                             name="finish = grid", hoverinfo="skip"))
    fig.update_layout(title="Grid vs finishing position (classified finishers)", xaxis_title="Grid position",
                      yaxis_title="Finishing position", yaxis_autorange="reversed")
    st.plotly_chart(style_fig(fig, 520), width="stretch")
    rho = cls[["grid", "finish_position"]].corr(method="spearman").iloc[0, 1]
    pole = data[data.grid == 1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Spearman grid ↔ finish", f"{rho:.2f}")
    c2.metric("Pole-sitter on the podium", f"{(pole[TARGET] == config.TOP3).mean():.0%}")
    c3.metric("Pole-sitter wins", f"{(pole.finish_position == 1).mean():.0%}")
    st.caption("Strong but far from perfect: the gap between grid and finish is what the model tries to explain.")

with tabs[1]:
    c1, c2 = st.columns([1, 1.6])
    with c1:
        counts = data[TARGET].value_counts().reindex(CLASS_ORDER).reset_index()
        counts.columns = ["category", "records"]
        fig = px.pie(counts, names="category", values="records", hole=.55, color="category",
                     color_discrete_map=CLASS_COLORS, title="Class balance")
        st.plotly_chart(style_fig(fig, 380), width="stretch")
    with c2:
        fig = px.bar(share(data, "season"), x="season", y="share", color=TARGET, category_orders={TARGET: CLASS_ORDER},
                     color_discrete_map=CLASS_COLORS, title="Category share by season")
        fig.update_layout(yaxis=dict(tickformat=".0%"), xaxis=dict(dtick=1))
        st.plotly_chart(style_fig(fig, 380), width="stretch")
    rows = []
    for s, part in data.groupby("season"):
        c = part[part.finish_position.notna()]
        rows.append({"season": s, "races": part.race_id.nunique(),
                     "Grid ↔ finish correlation": c[["grid", "finish_position"]].corr(method="spearman").iloc[0, 1],
                     "DNF/DSQ/DNS rate": (part.finish_status != "Classified").mean()})
    trend = pd.DataFrame(rows)
    metric = st.selectbox("Seasonal trend", ["Grid ↔ finish correlation", "DNF/DSQ/DNS rate"])
    fig = px.line(trend, x="season", y=metric, markers=True, color_discrete_sequence=[RED], hover_data=["races"])
    fig.update_layout(xaxis=dict(dtick=1), yaxis_title=None)
    st.plotly_chart(style_fig(fig, 340), width="stretch")

with tabs[2]:
    st.markdown("How strongly does each of the 20 model inputs relate to the finishing category "
                "(Spearman rank correlation with the ordered target Top 3 < Points < Outside Points)? "
                "Computed on the **training seasons** only.")
    corr = load_csv("eda_correlation_with_target.csv", index_col=0)
    if corr is not None:
        c = corr.reset_index().rename(columns={"index": "feature"})
        c["abs"] = c.spearman_with_target.abs()
        c = c.sort_values("abs")
        fig = px.bar(c, x="spearman_with_target", y="feature", orientation="h",
                     color=np.where(c.spearman_with_target >= 0, "positive", "negative"),
                     color_discrete_map={"positive": RED, "negative": "#3D85D1"},
                     title="Spearman correlation with finish category (train 2010–21)")
        fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title="ρ")
        st.plotly_chart(style_fig(fig, 620), width="stretch")
        st.caption("Positive ρ = higher value → worse category (e.g. a high grid number). Negative ρ = more points / "
                   "better team form → better category.")
    feats = all_race_features()
    train_ids = set(df[df.season.isin(config.TRAIN_SEASONS)].race_id)
    f = feats[feats.race_id.isin(train_ids)].merge(df[["race_id", "driver_id", TARGET]], on=["race_id", "driver_id"])
    pick = st.selectbox("Distribution of a feature by outcome", config.FEATURES,
                        format_func=lambda k: f"{k} — {FEATURE_INFO[k][0]}")
    fig = px.histogram(f, x=pick, color=TARGET, barmode="overlay", opacity=.65, nbins=40,
                       category_orders={TARGET: CLASS_ORDER}, color_discrete_map=CLASS_COLORS)
    st.plotly_chart(style_fig(fig, 380), width="stretch")

with tabs[3]:
    c1, c2 = st.columns(2)
    miss = load_csv("eda_missing_values.csv", index_col=0)
    out = load_csv("eda_outliers.csv")
    with c1:
        if miss is not None:
            m = miss.reset_index().rename(columns={"index": "feature"}).sort_values("pct_missing")
            fig = px.bar(m, x="pct_missing", y="feature", orientation="h", color_discrete_sequence=["#3D85D1"],
                         title="Missing values in model features (train, %)")
            fig.update_layout(yaxis_title=None, xaxis_title="% missing")
            st.plotly_chart(style_fig(fig, 460), width="stretch")
    with c2:
        if out is not None:
            o = out.sort_values("pct_outliers")
            fig = px.bar(o, x="pct_outliers", y="feature", orientation="h", color_discrete_sequence=[RED],
                         title="IQR outliers (% of train rows)")
            fig.update_layout(yaxis_title=None, xaxis_title="% outliers")
            st.plotly_chart(style_fig(fig, 460), width="stretch")
    st.markdown("**Handling:** missing values are filled with the training median *and* flagged with a missing-value "
                "indicator (e.g. first visit to a circuit has no circuit history). Outliers are real racing events "
                "(penalties, pit-lane starts, late-career drivers), so they are kept; the scaler is fitted on train only.")
    if out is not None:
        st.dataframe(out.round(2), width="stretch", hide_index=True)

with tabs[4]:
    level = st.radio("Group by", ["Driver", "Constructor", "Team lineage (rebrands merged)"], horizontal=True)
    col = {"Driver": "driver_name", "Constructor": "constructor_name",
           "Team lineage (rebrands merged)": "team_lineage"}[level]
    min_n = st.slider("Minimum entries", 10, 200, 60, step=10)
    agg = (data.groupby(col).agg(entries=("race_id", "count"), avg_grid=("grid", "mean"),
                                 avg_finish=("position_order", "mean"),
                                 podium_rate=(TARGET, lambda s: (s == config.TOP3).mean()),
                                 dnf_rate=("finish_status", lambda s: (s != "Classified").mean()))
           .query("entries >= @min_n").reset_index())
    fig = px.scatter(agg, x="avg_grid", y="avg_finish", size="entries", color="podium_rate", hover_name=col,
                     color_continuous_scale="Reds", title=f"Average grid vs average finish per {level.lower()}")
    fig.add_trace(go.Scatter(x=[1, 22], y=[1, 22], mode="lines", line=dict(color="#888", dash="dash"),
                             showlegend=False, hoverinfo="skip"))
    fig.update_layout(xaxis_title="Average grid", yaxis_title="Average finish position",
                      coloraxis_colorbar=dict(title="Podium rate", tickformat=".0%"))
    st.plotly_chart(style_fig(fig, 500), width="stretch")
    st.dataframe(agg.sort_values("avg_finish").assign(podium_rate=lambda d: d.podium_rate * 100,
                                                      dnf_rate=lambda d: d.dnf_rate * 100),
                 width="stretch", hide_index=True, column_config={
                     col: level, "avg_grid": st.column_config.NumberColumn("Avg grid", format="%.2f"),
                     "avg_finish": st.column_config.NumberColumn("Avg finish", format="%.2f"),
                     "podium_rate": st.column_config.NumberColumn("Podium %", format="%.0f%%"),
                     "dnf_rate": st.column_config.NumberColumn("DNF %", format="%.0f%%")})

with tabs[5]:
    n_races = data.groupby("circuit_name").race_id.nunique()
    keep = n_races[n_races >= 3].index
    t = share(data[data.circuit_name.isin(keep)], "circuit_name")
    order = t[t[TARGET] == config.OUTSIDE].sort_values("share").circuit_name.tolist()
    fig = px.bar(t, y="circuit_name", x="share", color=TARGET, orientation="h",
                 category_orders={TARGET: CLASS_ORDER, "circuit_name": order}, color_discrete_map=CLASS_COLORS,
                 title="Finishing categories by circuit (≥ 3 races)")
    fig.update_layout(xaxis=dict(tickformat=".0%", title="Share"), yaxis_title=None)
    st.plotly_chart(style_fig(fig, max(450, 22 * len(order))), width="stretch")
    cl = data[data.finish_position.notna()].assign(move=lambda d: (d.grid - d.finish_position).abs())
    mv = (cl.groupby("circuit_name").agg(mean_abs_move=("move", "mean"), races=("race_id", "nunique"))
          .query("races >= 3").sort_values("mean_abs_move", ascending=False).head(20).reset_index())
    fig = px.bar(mv, x="mean_abs_move", y="circuit_name", orientation="h", color_discrete_sequence=["#3D85D1"],
                 title="Circuits with the most position changes (mean |grid − finish|)")
    fig.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis_title="Places")
    st.plotly_chart(style_fig(fig, 520), width="stretch")

with tabs[6]:
    st.caption("Weather is collected for exploration only — it is **not** one of the model's 20 inputs.")
    w = data.groupby("race_id").agg(race_name=("race_name", "first"), temp=("temp_at_start_c", "first"),
                                    rain=("rain_before_start", "first"),
                                    dnf=("finish_status", lambda s: (s != "Classified").mean())).reset_index()
    w = w.dropna(subset=["temp"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Races with rain before start", int(w.rain.sum()))
    c2.metric("Median start temperature", f"{w.temp.median():.1f} °C")
    c3.metric("Races with weather data", len(w))
    a, b = st.columns(2)
    with a:
        fig = px.histogram(w, x="temp", nbins=25, color_discrete_sequence=[RED], title="Air temperature at race start")
        fig.update_layout(xaxis_title="°C", yaxis_title="Races")
        st.plotly_chart(style_fig(fig, 340), width="stretch")
    with b:
        w["conditions"] = np.where(w.rain == 1, "Rain before start", "Dry before start")
        fig = px.box(w, x="conditions", y="dnf", points="all", color="conditions", hover_name="race_name",
                     color_discrete_sequence=["#7A7F8C", "#3D85D1"], title="DNF/DSQ/DNS rate per race")
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis=dict(tickformat=".0%", title=None))
        st.plotly_chart(style_fig(fig, 340), width="stretch")

with tabs[7]:
    st.caption("Figures exported by the notebook into `reports/figures/`.")
    eda_figs = sorted(FIG.glob("eda_*.png"))
    labels = {p.name: p.stem.replace("eda_", "").replace("_", " ", 1).title() for p in eda_figs}
    pick = st.selectbox("Figure", [p.name for p in eda_figs], format_func=labels.get)
    if pick:
        st.image(str(FIG / pick), width="stretch")
