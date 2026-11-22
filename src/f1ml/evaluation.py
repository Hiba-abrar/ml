"""Metrics, confusion matrices and error analysis."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)

from f1ml import config

L = config.CLASS_ORDER


def summary_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=L, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=L, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    r = classification_report(y_true, y_pred, labels=L, output_dict=True, zero_division=0)
    return pd.DataFrame({c: r[c] for c in L}).T.rename(columns={"f1-score": "f1"})


def confusion_df(y_true, y_pred) -> pd.DataFrame:
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=L),
                        index=[f"actual {c}" for c in L], columns=[f"pred {c}" for c in L])


def roc_auc_ovr(y_true, proba, classes) -> float:
    """Macro one-vs-rest ROC-AUC; `proba` columns follow `classes` (sorted, as sklearn fits them)."""
    return roc_auc_score(y_true, np.asarray(proba), multi_class="ovr", average="macro", labels=list(classes))


def error_distance(y_true, y_pred) -> dict:
    idx = {c: i for i, c in enumerate(L)}
    d = np.abs(pd.Series(y_true).map(idx).to_numpy() - pd.Series(y_pred).map(idx).to_numpy())
    wrong = int((d > 0).sum())
    return {"errors": wrong, "adjacent_class_errors": int((d == 1).sum()),
            "top3_vs_outside_errors": int((d == 2).sum()),
            "adjacent_share_of_errors": float((d == 1).sum() / wrong) if wrong else float("nan")}
