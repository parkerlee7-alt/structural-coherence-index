"""
SCI Score v3 — Canonical Function
===================================
Structural Coherence Index: measures excess Hilbert-envelope autocorrelation
relative to a phase-randomized surrogate baseline that preserves the power spectrum.

Author: Parker J. Lee
Version: 3 (full diagnostic output)
Patent: US Provisional 63/904,444

Core scientific claim
---------------------
gap = c_obs - c_surr_mean

gap quantifies nonlinear amplitude organization beyond spectral prediction.
A positive gap means the signal's amplitude envelope contains temporal structure
that phase-randomized surrogates (which preserve the power spectrum exactly)
do not produce. This is the scientific object. SCI is a normalized display
of gap via a logistic function.

Version history
---------------
v1 — original, no seed (used in bearing paper, Zenodo May 2026)
v2 — seed=42 added (used in Sleep-EDF paper, 26 subjects)
v3 — full diagnostic dict returned, backwards-compatible scalar wrapper

Locked parameters (W=12, L=12, S=40, k=0.9, seed=42)
Do not change for pre-registered work. New parameters → new version label.

Usage
-----
    from sci_score_v3 import sci_score_v3, sci_score_v3_scalar

    result = sci_score_v3(signal)
    # result is a dict:
    # {
    #   'c_obs':       float,   envelope ACF of the real signal
    #   'c_surr_mean': float,   mean envelope ACF of phase-randomized surrogates
    #   'c_surr_std':  float,   std of surrogate distribution
    #   'gap':         float,   c_obs - c_surr_mean  (THE scientific object)
    #   'z':           float,   gap / c_surr_std, clipped to [-6, 6]
    #   'SCI':         float,   logistic(z * k), in [0, 1]
    #   'bucket':      str,     CORE / CORE_MID / TACTICAL / INELIGIBLE
    # }

    sci = sci_score_v3_scalar(signal)   # backwards-compatible: returns SCI only

Dependencies
------------
    pip install numpy scipy
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

# ──────────────────────────────────────────
# Locked parameters
# ──────────────────────────────────────────

_W    = 12     # envelope smoothing window
_L    = 12     # ACF lag count
_S    = 40     # surrogates
_K    = 0.9    # logistic slope
_SEED = 42     # RNG seed
_CLIP = 6.0    # z-score clip

_THRESH_CORE     = 0.75
_THRESH_CORE_MID = 0.65
_THRESH_TACTICAL = 0.55


# ──────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────

def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def _mean_acf(x: np.ndarray, max_lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < max_lag + 5:
        return float("nan")
    x = x - np.mean(x)
    denom = np.sum(x * x)
    if denom <= 1e-12:
        return float("nan")
    vals = [np.sum(x[:-lag] * x[lag:]) / denom for lag in range(1, max_lag + 1)]
    return float(np.mean(vals))


def _envelope(x: np.ndarray, w: int) -> np.ndarray:
    return _moving_average(np.abs(hilbert(x)), w)


def _phase_randomized_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=X.shape)
    phases[0] = 0.0
    if n % 2 == 0 and len(phases) > 1:
        phases[-1] = 0.0
    xs = np.fft.irfft(mag * np.exp(1j * phases), n=n)
    xs -= np.mean(xs)
    sd = np.std(xs)
    if sd > 1e-12:
        xs /= sd
    return xs


def _classify(sci: float) -> str:
    if not np.isfinite(sci):
        return "INELIGIBLE"
    if sci >= _THRESH_CORE:
        return "CORE"
    if sci >= _THRESH_CORE_MID:
        return "CORE_MID"
    if sci >= _THRESH_TACTICAL:
        return "TACTICAL"
    return "INELIGIBLE"


def _empty() -> dict:
    return {
        "c_obs": float("nan"),
        "c_surr_mean": float("nan"),
        "c_surr_std": float("nan"),
        "gap": float("nan"),
        "z": float("nan"),
        "SCI": float("nan"),
        "bucket": "INELIGIBLE",
    }


# ──────────────────────────────────────────
# Public API
# ──────────────────────────────────────────

def sci_score_v3(
    signal: np.ndarray,
    *,
    w: int = _W,
    L: int = _L,
    S: int = _S,
    k: float = _K,
    seed: int = _SEED,
    z_clip: float = _CLIP,
) -> dict:
    """
    Compute SCI v3 full diagnostic stack.

    Parameters
    ----------
    signal : array-like
        1-D signal. Will be mean-subtracted and standardized internally.
        NaN and Inf values are removed before processing.
    w, L, S, k, seed, z_clip
        SCI operator parameters. Defaults are the locked values.
        Pass explicitly only when you have a domain-specific reason.

    Returns
    -------
    dict with keys: c_obs, c_surr_mean, c_surr_std, gap, z, SCI, bucket
    """
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < L + 20 or np.std(x) <= 1e-12:
        return _empty()

    x = x - np.mean(x)
    sd = np.std(x)
    if sd > 1e-12:
        x = x / sd

    env = _envelope(x, w)
    c_obs = _mean_acf(env, L)
    if not np.isfinite(c_obs):
        return _empty()

    rng = np.random.default_rng(seed)
    sur = []
    for _ in range(S):
        xs = _phase_randomized_surrogate(x, rng)
        cs = _mean_acf(_envelope(xs, w), L)
        if np.isfinite(cs):
            sur.append(cs)

    if len(sur) < max(5, S // 3):
        return _empty()

    mu  = float(np.mean(sur))
    std = float(np.std(sur, ddof=1))
    gap = float(c_obs - mu)

    if std <= 1e-12:
        z   = float("nan")
        sci = float("nan")
    else:
        z   = float(np.clip(gap / std, -z_clip, z_clip))
        sci = float(1.0 / (1.0 + np.exp(-k * z)))

    return {
        "c_obs":       float(c_obs),
        "c_surr_mean": mu,
        "c_surr_std":  std,
        "gap":         gap,
        "z":           z,
        "SCI":         sci,
        "bucket":      _classify(sci),
    }


def sci_score_v3_scalar(signal: np.ndarray, **kwargs) -> float:
    """
    Backwards-compatible wrapper. Returns SCI score only (float).
    Use sci_score_v3() for the full diagnostic dict.
    """
    return sci_score_v3(signal, **kwargs)["SCI"]


# ──────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # White noise → gap should be near zero, SCI low
    noise = rng.standard_normal(5000)
    r = sci_score_v3(noise)
    print(f"White noise:   gap={r['gap']:+.4f}  SCI={r['SCI']:.3f}  bucket={r['bucket']}")

    # GARCH(1,1) alpha=0.1 beta=0.8 → gap should be clearly positive
    alpha, beta, omega = 0.1, 0.8, 0.01
    n = 12000
    eps = rng.standard_normal(n + 2000)
    h = np.zeros(n + 2000)
    x = np.zeros(n + 2000)
    h[0] = omega / (1 - alpha - beta)
    x[0] = np.sqrt(h[0]) * eps[0]
    for t in range(1, n + 2000):
        h[t] = omega + alpha * x[t-1]**2 + beta * h[t-1]
        x[t] = np.sqrt(max(h[t], 1e-12)) * eps[t]
    garch = x[2000:]
    r = sci_score_v3(garch)
    print(f"GARCH(1,1):    gap={r['gap']:+.4f}  SCI={r['SCI']:.3f}  bucket={r['bucket']}")

    assert r["gap"] > 0.01, "GARCH gap should be clearly positive"
    assert r["SCI"] > 0.75, "GARCH SCI should be CORE"
    print("Self-test passed.")
