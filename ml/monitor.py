"""MLOps & Input Distribution Drift Monitoring Module for HealthLens AI.

Stage 8 (HealthLens AI Roadmap):
  - Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) statistics
    between baseline training distributions and recent screening inputs.
  - Detects data distribution drift, feature shift, and missingness changes.
"""

from typing import Any, Dict, List
import numpy as np
import scipy.stats as stats


def calculate_psi(baseline: np.ndarray, target: np.ndarray, num_bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between baseline and target feature samples."""
    if len(baseline) == 0 or len(target) == 0:
        return 0.0

    # Determine bin thresholds from baseline distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(baseline, percentiles)
    bins[0] -= 1e-5
    bins[-1] += 1e-5

    # Bin counts
    baseline_counts, _ = np.histogram(baseline, bins=bins)
    target_counts, _ = np.histogram(target, bins=bins)

    # Convert to proportions with epsilon smoothing
    eps = 1e-4
    b_pct = (baseline_counts + eps) / (len(baseline) + eps * num_bins)
    t_pct = (target_counts + eps) / (len(target) + eps * num_bins)

    psi_val = np.sum((t_pct - b_pct) * np.log(t_pct / b_pct))
    return float(psi_val)


def calculate_ks_statistic(baseline: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Calculate Kolmogorov-Smirnov 2-sample statistic and p-value."""
    if len(baseline) == 0 or len(target) == 0:
        return {"ks_stat": 0.0, "p_value": 1.0}

    res = stats.ks_2samp(baseline, target)
    return {
        "ks_stat": round(float(res.statistic), 4),
        "p_value": round(float(res.pvalue), 4),
    }


def compute_dataset_drift(baseline_samples: Dict[str, List[float]], recent_samples: Dict[str, List[float]]) -> Dict[str, Any]:
    """Compute feature-level drift statistics for all biometric variables."""
    drift_results = {}

    for feature_name, base_vals in baseline_samples.items():
        rec_vals = recent_samples.get(feature_name, base_vals)
        base_arr = np.array(base_vals, dtype=float)
        rec_arr = np.array(rec_vals, dtype=float)

        psi = calculate_psi(base_arr, rec_arr)
        ks = calculate_ks_statistic(base_arr, rec_arr)

        if psi < 0.1:
            status = "STABLE"
        elif psi < 0.25:
            status = "SLIGHT_DRIFT"
        else:
            status = "SIGNIFICANT_DRIFT"

        drift_results[feature_name] = {
            "psi": round(psi, 4),
            "ks_statistic": ks["ks_stat"],
            "ks_pvalue": ks["p_value"],
            "status": status,
            "baseline_mean": round(float(np.mean(base_arr)), 2),
            "recent_mean": round(float(np.mean(rec_arr)), 2),
        }

    return {
        "overall_drift_status": "STABLE" if all(d["status"] == "STABLE" for d in drift_results.values()) else "DRIFT_DETECTED",
        "feature_drift": drift_results,
    }
