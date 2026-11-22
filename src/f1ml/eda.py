"""EDA figures. Computed on the TRAINING seasons only (class balance also shows the split sizes)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from f1ml import config

C = config.CLASS_COLORS
INK, MUTED = "#0b0b0b", "#52514e"


def style():
    sns.set_theme(style="whitegrid", rc={
        "axes.edgecolor": "#c9c8c3", "grid.color": "#e7e6e2", "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titleweight": "bold",
        "axes.titlesize": 12, "figure.facecolor": "white", "axes.facecolor": "white"})


def _save(fig, name, show=True):
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, dpi=150, bbox_inches="tight")
    return fig


def class_balance(parts, n_test):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    train = parts["train"]
    cnt = train[config.TARGET].value_counts().reindex(config.CLASS_ORDER)
    bars = ax[0].bar(cnt.index, cnt.values, color=[C[c] for c in cnt.index], width=0.6)
    for b, v in zip(bars, cnt.values):
        ax[0].text(b.get_x() + b.get_width() / 2, v + 40, f"{v:,}\n({v / cnt.sum():.0%})", ha="center", fontsize=9)
    ax[0].set_title("Class balance (training seasons)"); ax[0].set_ylabel("driver-race records")
    ax[0].set_ylim(0, cnt.max() * 1.2)
    share = pd.DataFrame({k: parts[k][config.TARGET].value_counts(normalize=True) for k in ("train", "validation")}
                         ).reindex(config.CLASS_ORDER).T
    share.plot(kind="bar", stacked=True, ax=ax[1], color=[C[c] for c in config.CLASS_ORDER],
               rot=0, edgecolor="white", width=0.6)
    ax[1].set_title(f"Class share: train vs validation (test set n={n_test:,} not inspected)"); ax[1].set_ylabel("share")
    ax[1].set_xticklabels([f"{k}\n(n={len(parts[k]):,})" for k in share.index])
    ax[1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1, 1))
    return _save(fig, "eda_1_class_balance.png")


def distributions(feats):
    cols = ["grid", "quali_gap_to_pole_pct", "driver_age", "driver_avg_finish_last5",
            "constructor_points_last5", "driver_points_before", "driver_dnf_rate_last5",
            "driver_career_races", "teammate_prev_finish", "driver_avg_quali_last5",
            "grid_penalty_places", "driver_standing_before"]
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    for ax, c in zip(axes.ravel(), cols):
        s = feats[c].dropna()
        ax.hist(s, bins=30, color=C[config.TOP3], edgecolor="white")
        ax.axvline(s.median(), color=C[config.POINTS], lw=1.5)
        ax.set_title(c, fontsize=10)
        ax.text(0.97, 0.9, f"skew {s.skew():.1f}", transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
    fig.suptitle("Feature distributions (training seasons; orange line = median)", y=1.01, fontweight="bold")
    return _save(fig, "eda_2_distributions.png")


def missing_values(feats):
    miss = (feats.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(miss) + 1.2))
    ax.barh(miss.index[::-1], miss.values[::-1], color=C[config.POINTS])
    for i, v in enumerate(miss.values[::-1]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("% missing"); ax.set_title("Missing values per feature (training seasons)")
    ax.set_xlim(0, miss.max() * 1.15)
    _save(fig, "eda_3_missing_values.png")
    return miss.rename("pct_missing").to_frame()


def outliers(feats):
    cols = ["quali_gap_to_pole_pct", "driver_age", "grid_penalty_places", "driver_avg_finish_last5",
            "constructor_points_last5", "driver_points_before"]
    rows = []
    fig, axes = plt.subplots(1, len(cols), figsize=(15, 3.8))
    for ax, c in zip(axes, cols):
        s = feats[c].dropna()
        q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({"feature": c, "n": len(s), "iqr_outliers": int(((s < lo) | (s > hi)).sum()),
                     "pct_outliers": round(100 * ((s < lo) | (s > hi)).mean(), 2),
                     "min": s.min(), "max": s.max()})
        ax.boxplot(s, patch_artist=True, boxprops=dict(facecolor="#cfe0f5"),
                   medianprops=dict(color=C[config.POINTS], lw=2),
                   flierprops=dict(marker=".", markersize=3, markerfacecolor=MUTED, alpha=.5))
        ax.set_title(c, fontsize=9); ax.set_xticks([])
    fig.suptitle("Outlier check: boxplots (points beyond 1.5 x IQR are outliers)", y=1.03, fontweight="bold")
    _save(fig, "eda_4_outliers.png")
    return pd.DataFrame(rows)


def correlations(feats, target_ordinal):
    d = feats.copy()
    d["finish_class (1=Top3,2=Points,3=Outside)"] = target_ordinal
    corr = d.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, square=True,
                cbar_kws={"shrink": .7}, linewidths=.3, linecolor="white")
    ax.set_title("Spearman correlation: features and target (training seasons)")
    _save(fig, "eda_5_correlations.png")
    return corr.iloc[:-1, -1].sort_values(key=abs, ascending=False).rename("spearman_with_target")


def grid_vs_class(train):
    t = train[train.grid <= 22]
    share = pd.crosstab(t.grid, t[config.TARGET], normalize="index")[config.CLASS_ORDER]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for c in config.CLASS_ORDER:
        ax[0].plot(share.index, share[c], marker="o", ms=4, lw=2, color=C[c], label=c)
    ax[0].set_xlabel("grid position"); ax[0].set_ylabel("probability")
    ax[0].set_title("Outcome class by starting grid slot"); ax[0].legend(frameon=False)
    cl = t[t.finish_position.notna()]
    sns.boxplot(data=cl, x="grid", y="finish_position", color="#cfe0f5", fliersize=1.5, ax=ax[1])
    ax[1].plot(range(0, 22), range(1, 23), color=C[config.POINTS], lw=1, ls="--", label="finish = grid")
    ax[1].invert_yaxis(); ax[1].set_title("Grid vs finishing position (classified finishers)")
    ax[1].legend(frameon=False)
    return _save(fig, "eda_6_grid_vs_outcome.png")


def class_by_season(df):
    sh = pd.crosstab(df.season, df[config.TARGET], normalize="index")[config.CLASS_ORDER]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    sh.plot(kind="bar", stacked=True, ax=ax, color=[C[c] for c in config.CLASS_ORDER], edgecolor="white", width=.8)
    ax.set_title("Outcome class share by season (all seasons)"); ax.set_ylabel("share")
    ax.legend(frameon=False, bbox_to_anchor=(1, 1))
    return _save(fig, "eda_7_class_by_season.png")
