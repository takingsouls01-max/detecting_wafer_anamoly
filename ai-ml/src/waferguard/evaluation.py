"""Evaluation metrics and reporting."""

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, roc_curve
)
from typing import Dict, Tuple
import json


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels [N]
        y_pred: Binary predictions [N]
        y_probs: Predicted probabilities [N]
        threshold: Decision threshold used

    Returns:
        Dictionary of metrics
    """
    # Classification metrics
    f1 = f1_score(y_true, y_pred, zero_division=0.0)
    precision = precision_score(y_true, y_pred, zero_division=0.0)
    recall = recall_score(y_true, y_pred, zero_division=0.0)

    # Ranking metrics (handle single-class case)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            roc_auc = roc_auc_score(y_true, y_probs)
            # Check for NaN (undefined ROC-AUC)
            if np.isnan(roc_auc):
                roc_auc = None
        except (ValueError, Exception):
            # Single class present - ROC-AUC undefined
            roc_auc = None

        try:
            avg_precision = average_precision_score(y_true, y_probs)
            # Check for NaN (undefined AP)
            if np.isnan(avg_precision):
                avg_precision = None
        except (ValueError, Exception):
            # No positive class - AP undefined
            avg_precision = None

    # Confusion matrix (explicitly specify labels to handle single-class case)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Derived metrics
    total = len(y_true)
    defect_count = y_true.sum()
    defect_rate = defect_count / total

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Per 1000 wafers metrics (more intuitive for operators)
    fp_per_1000 = (fp / total) * 1000
    fn_per_1000 = (fn / total) * 1000

    return {
        # Primary metrics
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'average_precision': float(avg_precision) if avg_precision is not None else None,

        # Confusion matrix
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),

        # Rates
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr),
        'defect_rate': float(defect_rate),

        # Per-1000 metrics
        'false_alarms_per_1000': float(fp_per_1000),
        'missed_defects_per_1000': float(fn_per_1000),

        # Sample counts
        'total_samples': int(total),
        'total_defects': int(defect_count),
        'predicted_defects': int(y_pred.sum()),

        # Configuration
        'threshold': float(threshold),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal threshold by maximizing validation metric.

    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        best_threshold, best_score
    """
    # Generate threshold candidates
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_probs)

    # Compute F1 for each threshold
    with np.errstate(divide='ignore', invalid='ignore'):
        if metric == 'f1':
            scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals)
        elif metric == 'precision':
            scores = precision_vals
        elif metric == 'recall':
            scores = recall_vals
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # Handle NaN values
    scores = np.nan_to_num(scores)

    # Find best threshold
    best_idx = np.argmax(scores)

    # Handle edge case where best_idx is at the end
    if best_idx >= len(thresholds):
        best_threshold = thresholds[-1] if len(thresholds) > 0 else 0.5
    else:
        best_threshold = thresholds[best_idx]

    best_score = scores[best_idx]

    return float(best_threshold), float(best_score)


def subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    legacy_mask: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Dict]:
    """
    Compute metrics separately for modern vs legacy wafers.

    Args:
        y_true: True labels
        y_pred: Predictions
        y_probs: Probabilities
        legacy_mask: Boolean mask indicating legacy wafers
        threshold: Decision threshold

    Returns:
        Dictionary with 'modern' and 'legacy' subgroup metrics
    """
    modern_mask = ~legacy_mask

    results = {}

    for name, mask in [('modern', modern_mask), ('legacy', legacy_mask)]:
        if mask.sum() == 0:
            results[name] = {'sample_count': 0}
            continue

        metrics = compute_metrics(
            y_true[mask],
            y_pred[mask],
            y_probs[mask],
            threshold
        )
        results[name] = metrics

    return results


def save_metrics(metrics: Dict, filepath: str):
    """Save metrics to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(filepath: str) -> Dict:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


class MetricsReport:
    """Structured metrics report for comparison."""

    def __init__(self, name: str, metrics: Dict):
        self.name = name
        self.metrics = metrics

    def __str__(self):
        m = self.metrics
        report = [
            f"=== {self.name} ===",
            f"Samples: {m['total_samples']} (defect rate: {m['defect_rate']:.2%})",
            f"",
            f"Classification Performance:",
            f"  F1 Score:      {m['f1_score']:.4f}",
            f"  Precision:     {m['precision']:.4f}",
            f"  Recall:        {m['recall']:.4f}",
            f"  ROC-AUC:       {m['roc_auc']:.4f}",
            f"  Avg Precision: {m['average_precision']:.4f}",
            f"",
            f"Confusion Matrix:",
            f"  TN={m['true_negatives']:4d}  FP={m['false_positives']:4d}",
            f"  FN={m['false_negatives']:4d}  TP={m['true_positives']:4d}",
            f"",
            f"Error Rates:",
            f"  False alarms: {m['false_alarms_per_1000']:.1f} per 1000 wafers",
            f"  Missed defects: {m['missed_defects_per_1000']:.1f} per 1000 wafers",
            f"",
            f"Threshold: {m['threshold']:.3f}",
        ]
        return "\n".join(report)
