import numpy as np, pandas as pd, pytest
from f1ml import config, data, modelling
from f1ml.features import PreRaceFeatureBuilder


@pytest.fixture(scope="module")
def parts():
    return data.split_data(data.load_data())


def test_split_is_chronological_and_complete(parts):
    assert len(parts["train"]) + len(parts["validation"]) + len(parts["test"]) == len(data.load_data())
    assert set(parts["train"].season) == set(config.TRAIN_SEASONS)
    assert parts["train"].race_id.isin(parts["test"].race_id).sum() == 0


def test_cv_folds_are_forward_in_time_and_race_grouped(parts):
    tr = parts["train"]
    for a, b in data.race_grouped_time_series_cv(tr):
        assert not set(tr.race_id.iloc[a]) & set(tr.race_id.iloc[b])
        assert tr.race_date.iloc[a].max() < tr.race_date.iloc[b].min()


def test_features_are_the_20_expected_and_no_outcome_columns(parts):
    f = PreRaceFeatureBuilder().fit(data.raw_inputs(parts["train"])).transform(data.raw_inputs(parts["train"]))
    assert list(f.columns) == config.FEATURES and len(config.FEATURES) == 20
    assert "teammate_prev_finish" in f.columns
    assert not {"position_order", "finish_position", "points", "finish_category", "finish_status"} & set(f.columns)


def test_preprocessing_fitted_on_training_rows_only(parts):
    X, y = data.raw_inputs(parts["train"]), parts["train"][config.TARGET]
    p = modelling.final_random_forest().fit(X, y)
    imp = p.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    feats = p["features"].transform(X)
    assert np.allclose(imp.statistics_, np.nanmedian(feats.to_numpy(float), axis=0), equal_nan=True)
    assert p.named_steps["prep"].named_transformers_["num"].named_steps["scale"].n_samples_seen_ == len(X)


def test_final_model_hyperparameters():
    clf = modelling.final_random_forest().named_steps["clf"]
    for k, v in config.FINAL_RF_PARAMS.items():
        assert getattr(clf, k) == v
    assert clf.n_estimators == 400 and clf.max_depth == 6 and clf.min_samples_leaf == 40 and clf.max_features == 0.3
