"""
Statistical Tests for Model Comparison

Wilcoxon, Bonferroni correction, Cohen's d, and more.
"""


import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon


def wilcoxon_comparison(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "two-sided",
) -> dict:
    """Wilcoxon signed-rank test for paired samples.
    
    Tests whether paired samples differ significantly.
    
    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B
        alternative: 'two-sided', 'less', or 'greater'
    
    Returns:
        Dictionary with statistic, p_value, significance
    """
    # Remove any NaN values
    mask = ~(np.isnan(scores_a) | np.isnan(scores_b))
    scores_a = scores_a[mask]
    scores_b = scores_b[mask]

    if len(scores_a) < 2:
        return {"error": "Insufficient samples"}

    try:
        statistic, p_value = wilcoxon(scores_a, scores_b, alternative=alternative)
        significant = p_value < 0.05

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": significant,
            "mean_a": float(np.mean(scores_a)),
            "mean_b": float(np.mean(scores_b)),
            "std_a": float(np.std(scores_a)),
            "std_b": float(np.std(scores_b)),
            "n": len(scores_a),
        }
    except Exception as e:
        return {"error": str(e)}


def mannwhitney_comparison(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "two-sided",
) -> dict:
    """Mann-Whitney U test for independent samples.
    
    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B
        alternative: 'two-sided', 'less', or 'greater'
    
    Returns:
        Dictionary with results
    """
    mask = ~(np.isnan(scores_a) | np.isnan(scores_b))
    scores_a = scores_a[mask]
    scores_b = scores_b[mask]

    if len(scores_a) < 2 or len(scores_b) < 2:
        return {"error": "Insufficient samples"}

    try:
        statistic, p_value = mannwhitneyu(scores_a, scores_b, alternative=alternative)
        significant = p_value < 0.05

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": significant,
            "mean_a": float(np.mean(scores_a)),
            "mean_b": float(np.mean(scores_b)),
            "n_a": len(scores_a),
            "n_b": len(scores_b),
        }
    except Exception as e:
        return {"error": str(e)}


def bonferroni_correct(
    p_values: list[float],
    alpha: float = 0.05,
) -> dict:
    """Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values
        alpha: Significance level
    
    Returns:
        Dictionary with corrected p-values and significance flags
    """
    n = len(p_values)
    corrected = [min(p * n, 1.0) for p in p_values]
    significant = [p < alpha for p in corrected]

    return {
        "original_p_values": p_values,
        "corrected_p_values": corrected,
        "significant": significant,
        "n_tests": n,
        "n_significant": sum(significant),
    }


def holm_bonferroni_correct(
    p_values: list[float],
    alpha: float = 0.05,
) -> dict:
    """Holm-Bonferroni correction (less conservative than Bonferroni).
    
    Args:
        p_values: List of p-values
        alpha: Significance level
    
    Returns:
        Dictionary with results
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    corrected = []
    significant = []

    for i, p in enumerate(sorted_p):
        corr_p = p * (n - i)
        corr_p = min(corr_p, 1.0)
        corrected.append(corr_p)
        significant.append(corr_p < alpha)

    # Restore original order
    corrected_original = [0] * n
    significant_original = [False] * n
    for idx, sorted_idx in enumerate(sorted_indices):
        corrected_original[sorted_idx] = corrected[idx]
        significant_original[sorted_idx] = significant[idx]

    return {
        "original_p_values": p_values,
        "corrected_p_values": corrected_original,
        "significant": significant_original,
        "n_tests": n,
        "n_significant": sum(significant_original),
    }


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
) -> float:
    """Cohen's d effect size.
    
    Args:
        group1: First group of values
        group2: Second group of values
    
    Returns:
        Cohen's d
    """
    mask = ~(np.isnan(group1) | np.isnan(group2))
    group1 = group1[mask]
    group2 = group2[mask]

    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    std1 = np.std(group1, ddof=1)
    std2 = np.std(group2, ddof=1)

    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std < 1e-8:
        return 0.0

    d = (mean1 - mean2) / pooled_std

    return float(d)


def effect_size_interpretation(d: float) -> str:
    """Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d
    
    Returns:
        Interpretation string
    """
    d = abs(d)

    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def friedman_test(
    *score_arrays: np.ndarray,
) -> dict:
    """Friedman test for comparing multiple models across multiple datasets.
    
    Args:
        *score_arrays: Variable number of score arrays
    
    Returns:
        Dictionary with test results
    """
    from scipy.stats import friedmanchisquare

    # Stack arrays
    data = np.vstack(score_arrays)

    if data.shape[0] < 2 or data.shape[1] < 2:
        return {"error": "Insufficient data"}

    try:
        stat, p_value = friedmanchisquare(*score_arrays)

        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "n_datasets": data.shape[1],
            "n_models": data.shape[0],
        }
    except Exception as e:
        return {"error": str(e)}
