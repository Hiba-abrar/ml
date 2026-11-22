"""Leakage-safe feature engineering as a scikit-learn transformer.

`PreRaceFeatureBuilder` is the FIRST step of every Pipeline. It receives the raw
race-entry table (one row per driver per race) and returns the 20 model inputs.

Why this is leakage-free
------------------------
* Every historical feature is a statistic over *strictly earlier* races
  (shift(1) inside driver / driver+circuit / team groups, or standings after the
  previous round only). A row's own outcome, and anything later, never enters
  its features. Outcome columns (position_order, is_dnf, weekend_points) are
  consumed only through those look-backs, so a race that has not happened yet
  can carry NaN outcomes and still be featurised.
* The teammate feature uses the teammate's previous race, never today's.
* The transformer learns no parameters. `fit` stores the history table (past
  results) and `transform` featurises new rows against history + the rows being
  transformed. In cross-validation, history = that fold's training races only.
* Everything that IS learned (imputation medians, scaling, the model) lives in
  later pipeline steps and sees training data only.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from f1ml import config

KEY = ["race_id", "driver_id"]
RAW_COLUMNS = [
    "race_id", "season", "round", "race_date", "driver_id", "constructor_id",
    "team_lineage", "circuit_id", "grid", "quali_position", "quali_gap_to_pole_pct",
    "date_of_birth", "pit_lane_start",
    # outcome columns: used ONLY through look-backs of earlier races
    "position_order", "is_dnf", "weekend_points",
]
WINDOW = 5
_CACHE: dict = {}          # content-hash -> feature frame (deterministic, stateless)


def _prev_window(s: pd.Series, how: str) -> pd.Series:
    r = s.shift(1).rolling(WINDOW, min_periods=1)
    return r.sum() if how == "sum" else r.mean()


def _standings_before_each_round(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    out = []
    er = df.groupby(["season", "round", "race_id", entity], as_index=False)["weekend_points"].sum()
    for _, sdf in er.groupby("season"):
        rounds = sorted(sdf["round"].unique())
        pts = (sdf.pivot_table(index=entity, columns="round", values="weekend_points", aggfunc="sum")
               .reindex(columns=rounds).fillna(0.0))
        cum = pts.cumsum(axis=1)
        first = sdf.groupby(entity)["round"].min()
        for i, rnd in enumerate(rounds):
            ent = sdf.loc[sdf["round"] == rnd, [entity, "race_id"]]
            if i == 0:
                standing = pd.Series(np.nan, index=ent[entity]); before = standing.copy()
            else:
                pool = first[first <= rnd].index
                p = cum.loc[pool, rounds[i - 1]]            # previous round only
                standing = p.rank(ascending=False, method="min").reindex(ent[entity])
                before = p.reindex(ent[entity])
            out.append(pd.DataFrame({"race_id": ent["race_id"].values, entity: ent[entity].values,
                                     "standing": standing.values, "points_before": before.values}))
    return pd.concat(out, ignore_index=True)


def compute_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Features for every row of `raw` (chronologically sortable), indexed by KEY."""
    df = raw.sort_values(["race_date", "round", "driver_id"]).reset_index(drop=True).copy()
    df["driver_age"] = (df["race_date"] - df["date_of_birth"]).dt.days / 365.25
    df["grid_penalty_places"] = df["grid"] - df["quali_position"]

    g = df.groupby("driver_id", sort=False)
    df["driver_prev_finish"] = g["position_order"].shift(1)
    df["driver_avg_finish_last5"] = g["position_order"].transform(lambda s: _prev_window(s, "mean"))
    df["driver_dnf_rate_last5"] = g["is_dnf"].transform(lambda s: _prev_window(s, "mean"))
    df["driver_avg_quali_last5"] = g["quali_position"].transform(lambda s: _prev_window(s, "mean"))
    df["driver_career_races"] = g.cumcount()

    gc = df.groupby(["driver_id", "circuit_id"], sort=False)
    df["driver_avg_finish_at_circuit"] = gc["position_order"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["driver_circuit_appearances"] = gc.cumcount()

    # teammate(s)' previous race result (mean over the other car(s) of the team)
    own = df["driver_prev_finish"]
    tg = df.groupby(["race_id", "constructor_id"])["driver_prev_finish"]
    n_other = tg.transform("count") - own.notna().astype(int)
    df["teammate_prev_finish"] = ((tg.transform("sum") - own.fillna(0)) / n_other).where(n_other > 0)

    # team points over the previous 5 weekends (aggregate per team-race, then shift)
    tr = (df.groupby(["team_lineage", "race_id"], as_index=False, sort=False)
          .agg(race_date=("race_date", "first"), round=("round", "first"),
               team_points=("weekend_points", "sum"))
          .sort_values(["race_date", "round"]).reset_index(drop=True))
    tr["constructor_points_last5"] = tr.groupby("team_lineage", sort=False)["team_points"].transform(
        lambda s: _prev_window(s, "sum"))
    df = df.merge(tr[["team_lineage", "race_id", "constructor_points_last5"]],
                  on=["team_lineage", "race_id"], how="left")

    d = _standings_before_each_round(df, "driver_id").rename(
        columns={"standing": "driver_standing_before", "points_before": "driver_points_before"})
    c = _standings_before_each_round(df, "constructor_id").rename(
        columns={"standing": "constructor_standing_before", "points_before": "constructor_points_before"})
    df = df.merge(d, on=["race_id", "driver_id"], how="left")
    df = df.merge(c, on=["race_id", "constructor_id"], how="left")
    return df.set_index(KEY)[config.FEATURES]


class PreRaceFeatureBuilder(BaseEstimator, TransformerMixin):
    """Raw race-entry table -> the 20 pre-race model inputs (DataFrame)."""

    def fit(self, X, y=None):
        X = self._check(X)
        self.history_ = X.copy()
        self.n_features_in_ = len(RAW_COLUMNS)
        return self

    def update_history(self, completed_rows: pd.DataFrame):
        """Append the results of races that have since been completed.

        Learns nothing: it only extends the look-up table of past results used to
        compute rolling statistics for later races.
        """
        new = self._check(completed_rows)
        h = pd.concat([self.history_, new], ignore_index=True)
        self.history_ = h.drop_duplicates(KEY, keep="last").reset_index(drop=True)
        return self

    def transform(self, X):
        X = self._check(X)
        keys = pd.MultiIndex.from_frame(X[KEY])
        hist = self.history_
        hist = hist[~pd.MultiIndex.from_frame(hist[KEY]).isin(keys)]
        ctx = pd.concat([hist, X], ignore_index=True)
        digest = hashlib.sha1(pd.util.hash_pandas_object(ctx, index=False).values.tobytes()).hexdigest()
        if digest not in _CACHE:
            if len(_CACHE) > 64:
                _CACHE.clear()
            _CACHE[digest] = compute_features(ctx)
        out = _CACHE[digest].reindex(keys)
        out.index = X.index
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(config.FEATURES, dtype=object)

    @staticmethod
    def _check(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PreRaceFeatureBuilder expects a pandas DataFrame of raw race entries")
        missing = [c for c in RAW_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"missing raw columns: {missing}")
        X = X[RAW_COLUMNS].copy()
        for column in ("race_id", "driver_id", "constructor_id", "team_lineage", "circuit_id"):
            X[column] = X[column].astype(object)
        X["race_date"] = pd.to_datetime(X["race_date"])
        X["date_of_birth"] = pd.to_datetime(X["date_of_birth"])
        if X.duplicated(KEY).any():
            raise ValueError("duplicate (race_id, driver_id) rows")
        return X
