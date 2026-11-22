# F1 finishing-category prediction (Top 3 / Points / Outside Points)

Predict, **before the race starts**, whether a driver finishes **Top 3**, in the **Points** (4–10) or **Outside Points** (11+/not classified).
Final model: **Random Forest** — 400 trees, `max_depth=6`, `min_samples_leaf=40`, `max_features=0.3`, `random_state=42`, no class weighting — on **20 features** including `teammate_prev_finish`.

## Results (final model, fitted on 2010–2021)
| | Accuracy | Macro-F1 |
|---|---|---|
| 5-fold CV (train seasons) | – | 0.678 ± 0.028 |
| Validation 2022–23 | 71.4 % | 70.1 % |
| **Test 2024–25 (reported once)** | **71.6 %** | **70.7 %** (ROC-AUC ovr 0.867) |
| Test, grid-rule baseline (reference) | 73.5 % | 72.0 % |

**Please read:** on the test seasons the trivial "finish where you start" rule scores ~2 points *higher* than the RF, although the RF was ~6 points better on 2022–23 validation. This is reported as found and discussed in the notebook (Sections 6–7).

## Requirement checklist
| # | Requirement | Where |
|---|---|---|
| 1 | EDA, ≥ 5 plots (distributions, missing values, outliers, correlations, class balance …) | notebook §2 → `reports/figures/eda_*.png` (7 figures) |
| 2 | Pre-processing (imputation, scaling, feature engineering) in a Pipeline fitted on train only | `src/f1ml/features.py`, `modelling.py`; notebook §3 (with leakage checks); `tests/` |
| 3 | Time-ordered train/validation/test split | 2010–21 / 2022–23 / 2024–25 (5,073 / 880 / 958 rows ≈ 73/13/14 %) – notebook §1 |
| 4 | Baselines + ≥ 3 algorithms | 3 baselines (majority, grid rule, grid-only LogReg) + Logistic Regression, KNN, SVM, Random Forest – §4 |
| 5 | 5-fold CV tuning | `GridSearchCV`, 5-fold expanding-window split over race weekends, macro-F1 – §4; `reports/results/cv_results_*.csv` |
| 6 | Metrics once on the test set; over/under-fitting and error discussion | §5–7; `reports/results/test_*.csv`, `final_results.json` |
| 7 | Final pipeline saved with joblib; pinned requirements | `models/final_pipeline.joblib`, `models/model_card.json`, `requirements.txt` |

## Quick start
```bash
python -m pip install -r requirements.txt
python run_all.py            # re-executes the whole notebook (~6 min, 2 CPU cores)
python -m pytest -q tests    # 8 tests: split, CV folds, leakage, fitted-on-train-only
python predict_example.py    # load the saved pipeline and predict a race
```
Open `notebooks/f1_final_project.ipynb` to read the executed analysis with all outputs.

## Streamlit app
```bash
python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py      # run from this folder
```
Seven pages: Overview · Data & EDA · Pipeline & Features · Model Lab · Test Evaluation · Race Predictor · Surprise Analysis.
The app only *reads* the saved artefacts (`models/`, `reports/results/`, `data/`) and the `f1ml` package; it never retrains.
The saved pipeline was built with scikit-learn 1.7.2 — keep that version pinned or loading will fail.

## How leakage is prevented
* Time-ordered split; test seasons are loaded but not used until one final cell.
* **One Pipeline:** `PreRaceFeatureBuilder` → `ColumnTransformer(median imputer + missing flags → StandardScaler)` → classifier. Imputer medians, scaler and trees are learnt on training rows only (inside CV: per training fold).
* Every history feature uses only strictly earlier races (`shift(1)`/previous-round standings); the teammate feature is the teammate's *previous* race. Tests alter a race's own and later outcomes and show its features do not change, and that an unplayed race (NaN outcomes) gets identical features.
* The feature step keeps a table of past results (`history_`) for rolling statistics. It learns nothing; it is extended with completed races before later races are predicted (`pipe["features"].update_history(rows)`).

## Provenance and verification
* `data/race_results_clean.csv` = cleaned Jolpica F1 API data (2010–2025, 6,911 rows). Rebuilding it from `data/interim/` with `src/f1ml/data_prep.py` reproduces it exactly. `data/data_sources.json` records sources/licences; the original API collector is in `provenance/original_data_collection.py.txt`.
* `provenance/frozen_reference/` holds your earlier frozen model and its saved validation/test predictions. The new pipeline, rebuilt from raw data, reproduces them **exactly**: 880/880 validation and 958/958 test predictions identical, probabilities equal to 1e-16, and all 20 features (incl. `teammate_prev_finish`) equal to the saved feature matrices.
* The specified RF config ranks 7th of 72 in the 5-fold CV grid (0.678 vs 0.681 for the best) — within CV noise; kept as the final model as specified (see notebook §4).

## Layout
```
notebooks/f1_final_project.ipynb   executed analysis (EDA → tuning → test → save)
src/f1ml/                          config, data, features (pipeline step), modelling, evaluation, eda, data_prep
models/                            final_pipeline.joblib, model_card.json
reports/figures/  reports/results/ plots and CSV/JSON results
tests/                             pytest suite
data/  provenance/                 data + frozen reference model and original collector
```
The saved pipeline pickles `f1ml` classes: run from this folder (or add `src/` to `PYTHONPATH`) when loading it.

## Limitations
Single fixed test split (958 rows, 48 races) — no confidence intervals; validation had been reused in earlier work, so it is not an independent estimate; qualifying/grid data are taken from the cached results, without a time-stamped publication history; pre-race information excludes weather forecasts, tyre strategy and practice pace.
