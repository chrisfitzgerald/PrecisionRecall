import numpy as np
import pandas as pd


def apply_threshold(confidence_scores: pd.Series, threshold: float) -> pd.Series:
    return (confidence_scores >= threshold).astype(int)


def compute_confusion_values(ground_truth: pd.Series, predictions: pd.Series) -> dict:
    tp = int(((ground_truth == 1) & (predictions == 1)).sum())
    fp = int(((ground_truth == 0) & (predictions == 1)).sum())
    fn = int(((ground_truth == 1) & (predictions == 0)).sum())
    tn = int(((ground_truth == 0) & (predictions == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_precision(tp: int, fp: int) -> float:
    if tp == 0 and fp == 0:
        return 1.0
    return tp / (tp + fp)


def compute_recall(tp: int, fn: int) -> float:
    if tp == 0 and fn == 0:
        return 0.0
    return tp / (tp + fn)


def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_pr_curve(
    ground_truth: pd.Series, confidence_scores: pd.Series
) -> tuple:
    thresholds = np.linspace(0.01, 0.99, 200)
    precisions = []
    recalls = []
    for t in thresholds:
        preds = apply_threshold(confidence_scores, t)
        cv = compute_confusion_values(ground_truth, preds)
        precisions.append(compute_precision(cv["tp"], cv["fp"]))
        recalls.append(compute_recall(cv["tp"], cv["fn"]))
    return np.array(precisions), np.array(recalls), thresholds


def find_f1_optimal_threshold(
    ground_truth: pd.Series, confidence_scores: pd.Series
) -> float:
    precisions, recalls, thresholds = compute_pr_curve(ground_truth, confidence_scores)
    f1_scores = np.array(
        [compute_f1(p, r) for p, r in zip(precisions, recalls)]
    )
    return float(thresholds[np.argmax(f1_scores)])
