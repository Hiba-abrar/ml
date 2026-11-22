"""Pipelines, baselines, candidate algorithms and hyper-parameter grids."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from f1ml import config
from f1ml.features import PreRaceFeatureBuilder


def build_preprocessor(columns) -> ColumnTransformer:
    """Median imputation (+ missing-value indicators) then standardisation.
    Fitted on training rows only because it is a Pipeline step."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([("num", numeric, list(columns))])


class GridRuleBaseline(BaseEstimator, ClassifierMixin):
    """'You finish where you start': grid 1-3 -> Top 3, 4-10 -> Points, 11+ -> Outside Points."""

    def fit(self, X, y=None):
        self.classes_ = np.array(config.CLASS_ORDER)
        return self

    def predict(self, X):
        g = pd.DataFrame(X)["grid"].to_numpy(dtype=float)
        return np.where(g <= 3, config.TOP3, np.where(g <= 10, config.POINTS, config.OUTSIDE))


def make_pipeline(clf, columns=None) -> Pipeline:
    """feature engineering -> imputation/scaling -> classifier. All steps in ONE Pipeline."""
    cols = columns or config.FEATURES
    return Pipeline([("features", PreRaceFeatureBuilder()),
                     ("prep", build_preprocessor(cols)),
                     ("clf", clf)])


def final_random_forest(n_jobs=2) -> Pipeline:
    return make_pipeline(RandomForestClassifier(**config.FINAL_RF_PARAMS, n_jobs=n_jobs))


def baselines() -> dict:
    rs = config.RANDOM_STATE
    return {
        "Baseline: majority class": make_pipeline(DummyClassifier(strategy="most_frequent")),
        "Baseline: grid rule": Pipeline([("features", PreRaceFeatureBuilder()),
                                         ("clf", GridRuleBaseline())]),
        "Baseline: grid-only LogReg": make_pipeline(LogisticRegression(max_iter=2000, random_state=rs), ["grid"]),
    }


def candidates() -> dict:
    """name -> (pipeline, GridSearchCV parameter grid). The final RF config is inside the RF grid."""
    rs = config.RANDOM_STATE
    return {
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=3000, random_state=rs)),
            {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]}),
        "KNN": (
            make_pipeline(KNeighborsClassifier()),
            {"clf__n_neighbors": [5, 15, 31, 51, 101], "clf__weights": ["uniform", "distance"]}),
        "SVM (RBF)": (
            make_pipeline(SVC(kernel="rbf", random_state=rs, cache_size=500)),
            {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01],
             "clf__class_weight": [None, "balanced"]}),
        "Random Forest": (
            make_pipeline(RandomForestClassifier(n_estimators=400, random_state=rs, n_jobs=1)),
            {"clf__max_depth": [4, 6, 8, 12], "clf__min_samples_leaf": [10, 25, 40],
             "clf__max_features": [0.3, 0.5, "sqrt"], "clf__class_weight": [None, "balanced"]}),
    }
