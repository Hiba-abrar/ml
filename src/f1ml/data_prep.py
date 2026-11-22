"""Stage 2 - Cleaning and preprocessing.

Takes the canonical tables in data/interim/ and produces ONE tidy table with
one row per driver per race (data/processed/race_results_clean.csv).

Main decisions (all documented in the README):
* Column names are snake_case, numeric columns are converted to numbers,
  dates to datetimes, identifiers to stripped strings.
* Duplicate (race, driver) rows are removed.
* Non-finishes are mapped from Ergast's positionText code, which is consistent
  across seasons (status strings are not):
      number -> classified finisher, R/N -> DNF, D/E -> DSQ, W -> DNS, F -> DNQ.
  A driver who retired late but was still *classified* (e.g. "+3 Laps" or
  "Accident" with a numeric positionText) keeps that classified position,
  because that is the official result (and points) they scored.
* DNQ rows (did not qualify, never started) are removed - they are not race
  participants. DNS/DNF/DSQ are kept and count as "Outside Points".
* grid = 0 means a pit-lane start: we set grid to the back of the grid
  (number of cars in that race) and add a pit_lane_start flag.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from f1ml import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """snake_case column names and strip whitespace from text columns."""
    df = df.copy()
    df.columns = (df.columns.astype(str).str.strip().str.replace(r"([a-z])([A-Z])", r"\1_\2", regex=True)
                  .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True).str.lower().str.strip("_"))
    for col in df.columns:
        inferred = pd.api.types.infer_dtype(df[col], skipna=True)
        if inferred in ("string", "mixed", "mixed-integer"):
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace({"": pd.NA, "\\N": pd.NA})
    return df


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def lap_time_to_seconds(value) -> float:
    """'1:29.708' -> 89.708 ; missing/invalid -> NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    try:
        if ":" in text:
            minutes, seconds = text.split(":", 1)
            return int(minutes) * 60 + float(seconds)
        return float(text)
    except ValueError:
        return np.nan


def classify_status(position_text) -> str:
    """Map Ergast positionText to a consistent finish status."""
    if pd.isna(position_text):
        return "DNF"
    text = str(position_text).strip().upper()
    if text.isdigit():
        return "Classified"
    return {"R": "DNF", "N": "DNF", "D": "DSQ", "E": "DSQ", "W": "DNS", "F": "DNQ"}.get(text, "DNF")


def finish_category_from_position(classified_position) -> str:
    """Target definition: 1-3 -> Top 3, 4-10 -> Points, everything else -> Outside Points.

    *classified_position* is NaN for DNF/DSQ/DNS, which therefore become
    "Outside Points" as required.
    """
    if pd.isna(classified_position):
        return config.OUTSIDE
    pos = int(classified_position)
    if pos <= 3:
        return config.TOP3
    if pos <= 10:
        return config.POINTS
    return config.OUTSIDE


# --------------------------------------------------------------------------
# Main cleaning function
# --------------------------------------------------------------------------
def load_interim_tables(interim_dir=None) -> dict[str, pd.DataFrame]:
    interim_dir = interim_dir or config.INTERIM_DIR
    names = ["races", "results", "qualifying", "sprint_results", "drivers", "constructors", "weather"]
    tables = {}
    for name in names:
        path = interim_dir / f"{name}.csv"
        if path.exists():
            tables[name] = pd.read_csv(path, dtype={"time_utc": "string"})
        elif name in ("weather", "sprint_results"):
            logger.warning("%s not found - continuing without it.", path.name)
            tables[name] = pd.DataFrame({"race_id": pd.Series(dtype="string")})
        else:
            raise FileNotFoundError(f"{path} missing - run data collection first "
                                    "(python run_pipeline.py --stage collect).")
    return tables


def build_clean_dataset(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Merge and clean the canonical tables. Returns (clean_df, quality_report)."""
    report: dict = {}
    races = standardize_columns(tables["races"])
    results = standardize_columns(tables["results"])
    quali = standardize_columns(tables["qualifying"])
    sprints = standardize_columns(tables["sprint_results"])
    drivers = standardize_columns(tables["drivers"])
    constructors = standardize_columns(tables["constructors"])
    weather = standardize_columns(tables["weather"])

    races = to_numeric(races, ["season", "round", "lat", "lon"])
    races["race_date"] = pd.to_datetime(races["date"], errors="coerce")
    results = to_numeric(results, ["season", "round", "grid", "position", "points", "laps", "car_number"])
    quali = to_numeric(quali, ["quali_position"])
    report["raw_result_rows"] = int(len(results))

    # ---- duplicates -------------------------------------------------------
    before = len(results)
    results = results.drop_duplicates(subset=["race_id", "driver_id"], keep="first")
    report["duplicate_result_rows_removed"] = int(before - len(results))
    races = races.drop_duplicates(subset=["race_id"])
    quali = (quali.sort_values("quali_position")
             .drop_duplicates(subset=["race_id", "driver_id"], keep="first"))
    drivers = drivers.drop_duplicates("driver_id")
    constructors = constructors.drop_duplicates("constructor_id")

    # ---- finish status and target ------------------------------------------
    results["finish_status"] = results["position_text"].map(classify_status)
    report["finish_status_counts_before_dnq_removal"] = results["finish_status"].value_counts().to_dict()
    results = results[results["finish_status"] != "DNQ"].copy()
    report["dnq_rows_removed"] = int(report["finish_status_counts_before_dnq_removal"].get("DNQ", 0))

    is_classified = results["finish_status"] == "Classified"
    results["finish_position"] = np.where(is_classified, results["position"], np.nan)
    results["is_dnf"] = (~is_classified).astype(int)       # DNF, DSQ or DNS
    # 'position_order' = official classification order for everybody (DNFs are
    # ordered by laps completed). Used for "average finishing position" history.
    results["position_order"] = results["position"]
    missing_order = results["position_order"].isna()
    if missing_order.any():
        results.loc[missing_order, "position_order"] = (
            results[missing_order].groupby("race_id").cumcount()
            + results.groupby("race_id")["position_order"].transform("max")[missing_order] + 1)
    results[config.TARGET] = results["finish_position"].map(finish_category_from_position)

    # ---- grid: pit-lane starts --------------------------------------------
    cars_in_race = results.groupby("race_id")["driver_id"].transform("count")
    results["pit_lane_start"] = (results["grid"].fillna(0) == 0).astype(int)
    results["grid"] = np.where(results["pit_lane_start"] == 1, cars_in_race, results["grid"])

    # ---- qualifying times -> gap to pole --------------------------------
    for col in ["q1", "q2", "q3"]:
        if col not in quali.columns:
            quali[col] = np.nan
        quali[f"{col}_s"] = quali[col].map(lap_time_to_seconds)
    quali["best_quali_time_s"] = quali[["q1_s", "q2_s", "q3_s"]].min(axis=1)
    pole_time = quali.groupby("race_id")["best_quali_time_s"].transform("min")
    # Percentage gap of the driver's fastest qualifying lap to the fastest lap of
    # the whole session. Qualifying is completed before the race -> no leakage.
    quali["quali_gap_to_pole_pct"] = (quali["best_quali_time_s"] - pole_time) / pole_time * 100
    quali = quali[["race_id", "driver_id", "quali_position", "best_quali_time_s",
                   "quali_gap_to_pole_pct"]]

    # ---- sprint points (only used later for HISTORICAL team/standing stats) ----
    if not sprints.empty and "sprint_points" in sprints.columns:
        sprints = to_numeric(sprints, ["sprint_points"])
        sprints = (sprints.groupby(["race_id", "driver_id"], as_index=False)["sprint_points"].sum())
    else:
        sprints = pd.DataFrame(columns=["race_id", "driver_id", "sprint_points"])

    # ---- merge everything --------------------------------------------------
    race_cols = ["race_id", "race_name", "race_date", "time_utc", "circuit_id", "circuit_name",
                 "locality", "country", "lat", "lon"]
    df = results.merge(races[race_cols], on="race_id", how="left")
    df = df.merge(quali, on=["race_id", "driver_id"], how="left")
    df = df.merge(sprints, on=["race_id", "driver_id"], how="left")
    df["sprint_points"] = pd.to_numeric(df["sprint_points"], errors="coerce").fillna(0.0)
    df["points"] = df["points"].fillna(0.0)
    df["weekend_points"] = df["points"] + df["sprint_points"]
    df = df.merge(drivers[["driver_id", "driver_code", "given_name", "family_name", "date_of_birth"]],
                  on="driver_id", how="left")
    df["driver_name"] = (df["given_name"].fillna("") + " " + df["family_name"].fillna("")).str.strip()
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df = df.merge(constructors[["constructor_id", "constructor_name"]], on="constructor_id", how="left")
    df["constructor_name"] = df["constructor_name"].fillna(df["constructor_id"])
    df["team_lineage"] = df["constructor_id"].map(lambda c: config.CONSTRUCTOR_LINEAGE.get(c, c))

    weather_cols = [c for c in ["race_id", "temp_at_start_c", "precip_3h_before_mm",
                                "rain_before_start", "precip_race_window_mm", "rain_during_race"]
                    if c in weather.columns]
    df = df.merge(weather[weather_cols], on="race_id", how="left")
    for col in ["temp_at_start_c", "precip_3h_before_mm", "rain_before_start",
                "precip_race_window_mm", "rain_during_race"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rows without a race date cannot be ordered chronologically -> drop.
    missing_dates = df["race_date"].isna().sum()
    df = df[df["race_date"].notna()]
    report["rows_dropped_missing_race_date"] = int(missing_dates)

    df = df.sort_values(["race_date", "round", "position_order"]).reset_index(drop=True)
    df["season"] = df["season"].astype(int)
    df["round"] = df["round"].astype(int)

    report["final_rows"] = int(len(df))
    report["races"] = int(df["race_id"].nunique())
    report["seasons"] = sorted(int(s) for s in df["season"].unique())
    report["drivers"] = int(df["driver_id"].nunique())
    report["constructors"] = int(df["constructor_id"].nunique())
    report["circuits"] = int(df["circuit_id"].nunique())
    report["target_distribution"] = df[config.TARGET].value_counts().to_dict()
    report["finish_status_counts"] = df["finish_status"].value_counts().to_dict()
    report["missing_values"] = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
    return df, report


def validate_standings(df: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    """Compare points summed from race+sprint results with OFFICIAL final standings.

    Only a data-quality check: confirms that the race-by-race points we use to
    derive pre-race standings add up to the official season totals.
    """
    rows = []
    for kind, key in (("driver", "driver_id"), ("constructor", "constructor_id")):
        derived = df.groupby(["season", key])["weekend_points"].sum().rename("derived_points").reset_index()
        derived = derived.rename(columns={key: "entity_id"})
        off = official[official["kind"] == kind][["season", "entity_id", "official_points"]]
        merged = derived.merge(off, on=["season", "entity_id"], how="inner")
        merged["kind"] = kind
        rows.append(merged)
    out = pd.concat(rows, ignore_index=True)
    out["official_points"] = pd.to_numeric(out["official_points"], errors="coerce")
    out["difference"] = out["derived_points"] - out["official_points"]
    return out


def run_preprocessing(out_path=None) -> pd.DataFrame:
    """Rebuild data/race_results_clean.csv from the interim tables."""
    tables = load_interim_tables()
    df, report = build_clean_dataset(tables)
    df.to_csv(out_path or config.CLEAN_FILE, index=False)
    logger.info("Clean dataset: %d rows, %d races", report["final_rows"], report["races"])
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_preprocessing()
