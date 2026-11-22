"""Loading and the time-ordered train / validation / test split."""
import numpy as np
import pandas as pd

from f1ml import config
from f1ml.features import RAW_COLUMNS


def load_data() -> pd.DataFrame:
    df = pd.read_csv(config.CLEAN_FILE, parse_dates=["race_date", "date_of_birth"])
    return df.sort_values(["race_date", "round", "driver_id"]).reset_index(drop=True)


def split_data(df: pd.DataFrame) -> dict:
    """Season-based chronological split: 2010-21 train | 2022-23 validation | 2024-25 test."""
    parts = {
        "train": df[df.season.isin(config.TRAIN_SEASONS)],
        "validation": df[df.season.isin(config.VALID_SEASONS)],
        "test": df[df.season.isin(config.TEST_SEASONS)],
    }
    parts = {k: v.reset_index(drop=True) for k, v in parts.items()}
    # chronological guarantees
    assert parts["train"].race_date.max() < parts["validation"].race_date.min()
    assert parts["validation"].race_date.max() < parts["test"].race_date.min()
    return parts


def raw_inputs(part: pd.DataFrame) -> pd.DataFrame:
    """The raw columns the pipeline consumes (no target, no post-race fields except as history)."""
    return part[RAW_COLUMNS]


def race_grouped_time_series_cv(df: pd.DataFrame, n_splits: int = config.N_CV_SPLITS):
    """Expanding-window 5-fold CV over RACE WEEKENDS (all drivers of a race stay together,
    every validation fold is later in time than its training fold)."""
    from sklearn.model_selection import TimeSeriesSplit
    races = (df[["race_id", "race_date", "round"]].drop_duplicates("race_id")
             .sort_values(["race_date", "round"])["race_id"].to_numpy())
    row_race = df["race_id"].to_numpy()
    return [(np.flatnonzero(np.isin(row_race, races[a])), np.flatnonzero(np.isin(row_race, races[b])))
            for a, b in TimeSeriesSplit(n_splits=n_splits).split(races)]
