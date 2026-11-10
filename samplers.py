"""Temperature, top-k, top-p and min-p, operating on one vector of logits. Numpy only."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by T. T below 1 sharpens the distribution, above 1 flattens it."""
    if temperature <= 0:
        raise ValueError("use greedy() for temperature 0")
    return logits / temperature


def top_k_mask(probs: np.ndarray, k: int) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be at least 1")
    k = min(k, probs.size)
    return probs >= np.partition(probs, -k)[-k]


def top_p_mask(probs: np.ndarray, p: float) -> np.ndarray:
    """Keep the smallest set of top tokens whose total probability reaches p."""
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    order = np.argsort(-probs)
    cumulative = np.cumsum(probs[order])
    keep = order[: int(np.searchsorted(cumulative, p)) + 1]
    mask = np.zeros(probs.size, dtype=bool)
    mask[keep] = True
    return mask


def min_p_mask(probs: np.ndarray, min_p: float) -> np.ndarray:
    """Keep tokens at least min_p times as likely as the single most likely token."""
    if not 0 <= min_p <= 1:
        raise ValueError("min_p must be in [0, 1]")
    return probs >= min_p * probs.max()


def filtered(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
) -> np.ndarray:
    """The distribution actually sampled from. Temperature first, then truncation."""
    probs = softmax(apply_temperature(logits, temperature))
    mask = np.ones(probs.size, dtype=bool)
    if top_k is not None:
        mask &= top_k_mask(probs, top_k)
    if top_p is not None:
        mask &= top_p_mask(probs, top_p)
    if min_p is not None:
        mask &= min_p_mask(probs, min_p)
    probs = np.where(mask, probs, 0.0)
    return probs / probs.sum()


def greedy(logits: np.ndarray) -> int:
    return int(np.argmax(logits))
