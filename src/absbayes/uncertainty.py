"""Pointwise uncertainty summaries for the 2026-minus-2025 zone surface."""
import numpy as np


def change_summary(s25, s26):
    """s25, s26: posterior draws of the surface on the grid, shape (draws, G).

    Returns the posterior mean change, the probability the change is above zero,
    and the 90% interval, each of shape (G,).
    """
    delta = np.asarray(s26) - np.asarray(s25)
    mean = delta.mean(axis=0)
    p_pos = (delta > 0).mean(axis=0)
    lo, hi = np.percentile(delta, [5, 95], axis=0)
    return mean, p_pos, lo, hi


def meaningful_change(s25, s26, threshold=0.05, level=0.90):
    """Cells where the change is likely large enough to matter, in a consistent direction.

    A cell is flagged when P(change > threshold) >= level or P(change < -threshold) >= level.
    This drops the far field, where both seasons' strike probabilities are near zero and a
    tiny difference can look "significant" only because it is measured precisely.
    Returns (mask, p_up, p_down), each of shape (G,).
    """
    delta = np.asarray(s26) - np.asarray(s25)
    p_up = (delta > threshold).mean(axis=0)
    p_down = (delta < -threshold).mean(axis=0)
    return (p_up >= level) | (p_down >= level), p_up, p_down
