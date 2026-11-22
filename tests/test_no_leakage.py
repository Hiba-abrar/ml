import numpy as np, pandas as pd
from f1ml import config, data
from f1ml.features import PreRaceFeatureBuilder


def _rows():
    df = data.load_data()
    return data.raw_inputs(df[df.season.between(2019, 2022)].reset_index(drop=True))


def test_features_ignore_own_and_future_outcomes():
    rows = _rows()
    race = "2021_10"
    base = PreRaceFeatureBuilder().fit(rows).transform(rows)
    t = rows.copy()
    m = (t.race_id >= race).to_numpy()
    t.loc[m, ["position_order", "is_dnf", "weekend_points"]] = [1, 0, 99]
    alt = PreRaceFeatureBuilder().fit(t).transform(t)
    cur = (rows.race_id == race).to_numpy()
    assert np.allclose(base[cur].to_numpy(float), alt[cur].to_numpy(float), equal_nan=True)
    early = (rows.race_id < race).to_numpy()
    assert np.allclose(base[early].to_numpy(float), alt[early].to_numpy(float), equal_nan=True)


def test_unplayed_race_with_nan_outcomes_gets_same_features():
    rows = _rows()
    race = "2022_05"
    full = PreRaceFeatureBuilder().fit(rows).transform(rows)
    past = rows[rows.race_id < race]
    future = rows[rows.race_id == race].copy()
    future[["position_order", "is_dnf", "weekend_points"]] = np.nan
    b = PreRaceFeatureBuilder().fit(past)
    got = b.transform(future)
    exp = full[(rows.race_id == race).to_numpy()]
    assert np.allclose(got.to_numpy(float), exp.to_numpy(float), equal_nan=True)


def test_teammate_feature_is_teammates_previous_race_not_current():
    rows = _rows()
    f = PreRaceFeatureBuilder().fit(rows).transform(rows)
    d = rows.assign(prev=rows.sort_values(["race_date", "driver_id"]).groupby("driver_id").position_order.shift(1)
                    .reindex(rows.index), tm=f.teammate_prev_finish.to_numpy())
    r = d[(d.race_id == "2021_10") & (d.constructor_id == "mercedes")]
    assert len(r) == 2
    a, b = r.iloc[0], r.iloc[1]
    assert np.isclose(a.tm, b.prev) and np.isclose(b.tm, a.prev)
