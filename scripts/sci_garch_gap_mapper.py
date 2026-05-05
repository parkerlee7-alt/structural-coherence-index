#!/usr/bin/env python3
"""
SCI GARCH Gap Mapper
Author: Parker J. Lee / generated with ChatGPT

Purpose
-------
Empirically test the revised SCI derivation target:

    Nonlinear amplitude dynamics, especially GARCH-type volatility persistence,
    produce envelope-coherence excess beyond phase-randomized spectral surrogates.

This script sweeps GARCH(1,1) parameters and measures:

    1. c_obs       = observed Hilbert-envelope autocorrelation coherence
    2. c_surr_mean = mean surrogate envelope coherence
    3. gap         = c_obs - c_surr_mean
    4. z           = normalized surrogate excess
    5. SCI         = logistic(z)
    6. amp_acf     = autocorrelation of |x| or x^2 as a proxy for volatility persistence

The key empirical question:

    Does gap increase as alpha + beta approaches 1?

Dependencies
------------
pip install numpy scipy pandas matplotlib

Run
---
python sci_garch_gap_mapper.py

Outputs
-------
Creates an output folder:

    sci_garch_gap_results/

with:
    garch_gap_results.csv
    gap_vs_persistence.png
    sci_vs_persistence.png
    heatmap_gap_alpha_beta.png
    heatmap_sci_alpha_beta.png
"""

import os
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import pearsonr


# -----------------------------
# Core SCI parameters
# -----------------------------

@dataclass
class SCIParams:
    n_samples: int = 10_000
    burn_in: int = 2_000
    envelope_smoothing_w: int = 12
    acf_lags_L: int = 12
    surrogates_S: int = 40
    logistic_k: float = 0.9
    z_clip: float = 6.0
    random_seed: int = 42


# -----------------------------
# GARCH simulation
# -----------------------------

def simulate_garch_11(
    n: int,
    alpha: float,
    beta: float,
    omega: float = 0.01,
    burn_in: int = 2_000,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate stationary GARCH(1,1):

        x_t = sigma_t * eps_t
        sigma_t^2 = omega + alpha*x_{t-1}^2 + beta*sigma_{t-1}^2

    Stationarity requires alpha + beta < 1.
    """
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be nonnegative.")
    if alpha + beta >= 1:
        raise ValueError("GARCH stationarity requires alpha + beta < 1.")

    rng = np.random.default_rng(seed)
    total_n = n + burn_in

    eps = rng.normal(0, 1, total_n)
    x = np.zeros(total_n)
    sigma2 = np.zeros(total_n)

    # unconditional variance
    sigma2[0] = omega / max(1e-12, (1.0 - alpha - beta))
    x[0] = math.sqrt(sigma2[0]) * eps[0]

    for t in range(1, total_n):
        sigma2[t] = omega + alpha * (x[t - 1] ** 2) + beta * sigma2[t - 1]
        x[t] = math.sqrt(max(sigma2[t], 1e-12)) * eps[t]

    x = x[burn_in:]

    # Standardize so scale does not dominate comparisons.
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)

    return x


# -----------------------------
# Comparison signals
# -----------------------------

def simulate_ar1(n: int, phi: float = 0.95, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 1, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    return x


def simulate_white_noise(n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


# -----------------------------
# SCI computation
# -----------------------------

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def mean_acf(x: np.ndarray, max_lag: int) -> float:
    """
    Mean autocorrelation from lag 1 to max_lag.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    denom = np.sum(x ** 2) + 1e-12

    vals = []
    for lag in range(1, max_lag + 1):
        if lag >= len(x):
            break
        vals.append(np.sum(x[:-lag] * x[lag:]) / denom)

    if not vals:
        return np.nan

    return float(np.mean(vals))


def envelope_coherence(x: np.ndarray, w: int, L: int) -> float:
    """
    Hilbert amplitude envelope -> smoothing -> mean short-lag ACF.
    """
    z = hilbert(x)
    env = np.abs(z)
    env_smooth = moving_average(env, w)
    return mean_acf(env_smooth, L)


def phase_randomized_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Phase-randomized surrogate preserving exact spectral magnitude.
    Works for real-valued x using rFFT/irFFT.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    X = np.fft.rfft(x)
    mag = np.abs(X)

    # Random phases for positive frequencies.
    phases = rng.uniform(0, 2 * np.pi, len(X))

    # DC component phase must be zero.
    phases[0] = 0.0

    # Nyquist component for even n must also be real-valued.
    if n % 2 == 0:
        phases[-1] = 0.0

    Xs = mag * np.exp(1j * phases)
    xs = np.fft.irfft(Xs, n=n)

    xs = (xs - np.mean(xs)) / (np.std(xs) + 1e-12)
    return xs


def compute_sci_metrics(x: np.ndarray, params: SCIParams, seed: int | None = None) -> dict:
    """
    Compute observed coherence, surrogate coherence, gap, z, and SCI.
    """
    rng = np.random.default_rng(seed)

    c_obs = envelope_coherence(x, params.envelope_smoothing_w, params.acf_lags_L)

    surr_scores = []
    for _ in range(params.surrogates_S):
        xs = phase_randomized_surrogate(x, rng)
        surr_scores.append(
            envelope_coherence(xs, params.envelope_smoothing_w, params.acf_lags_L)
        )

    surr_scores = np.array(surr_scores, dtype=float)
    c_surr_mean = float(np.mean(surr_scores))
    c_surr_std = float(np.std(surr_scores, ddof=1) + 1e-12)

    gap = float(c_obs - c_surr_mean)
    z = gap / c_surr_std
    z = float(np.clip(z, -params.z_clip, params.z_clip))
    sci = float(1.0 / (1.0 + np.exp(-params.logistic_k * z)))

    # Extra amplitude-dynamics diagnostics.
    abs_acf = mean_acf(np.abs(x), params.acf_lags_L)
    sq_acf = mean_acf(x ** 2, params.acf_lags_L)

    return {
        "c_obs": c_obs,
        "c_surr_mean": c_surr_mean,
        "c_surr_std": c_surr_std,
        "gap": gap,
        "z": z,
        "SCI": sci,
        "abs_acf": abs_acf,
        "sq_acf": sq_acf,
    }


# -----------------------------
# Sweep experiment
# -----------------------------

def run_garch_sweep(params: SCIParams) -> pd.DataFrame:
    """
    Sweep alpha and beta values while enforcing alpha + beta < 1.
    """
    rows = []

    # Designed to cover weak, medium, and near-integrated volatility persistence.
    alphas = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    betas = [0.10, 0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.94, 0.97]

    run_id = 0

    for alpha in alphas:
        for beta in betas:
            persistence = alpha + beta
            if persistence >= 0.995:
                continue
            if persistence >= 1.0:
                continue

            run_id += 1
            seed = params.random_seed + run_id

            try:
                x = simulate_garch_11(
                    n=params.n_samples,
                    alpha=alpha,
                    beta=beta,
                    burn_in=params.burn_in,
                    seed=seed,
                )
                metrics = compute_sci_metrics(x, params, seed=seed + 10_000)

                rows.append({
                    "signal": "GARCH(1,1)",
                    "alpha": alpha,
                    "beta": beta,
                    "persistence_alpha_plus_beta": persistence,
                    **metrics,
                })

                print(
                    f"GARCH alpha={alpha:.3f}, beta={beta:.3f}, "
                    f"a+b={persistence:.3f}, gap={metrics['gap']:.4f}, SCI={metrics['SCI']:.3f}"
                )

            except Exception as e:
                print(f"Skipped alpha={alpha}, beta={beta}: {e}")

    return pd.DataFrame(rows)


def run_baselines(params: SCIParams) -> pd.DataFrame:
    """
    Add white noise and AR(1) controls.
    """
    rows = []

    baseline_specs = [
        ("White noise", lambda seed: simulate_white_noise(params.n_samples, seed=seed)),
        ("AR(1) phi=0.95", lambda seed: simulate_ar1(params.n_samples, phi=0.95, seed=seed)),
    ]

    for i, (name, fn) in enumerate(baseline_specs):
        seed = params.random_seed + 50_000 + i
        x = fn(seed)
        metrics = compute_sci_metrics(x, params, seed=seed + 1_000)

        rows.append({
            "signal": name,
            "alpha": np.nan,
            "beta": np.nan,
            "persistence_alpha_plus_beta": np.nan,
            **metrics,
        })

        print(f"{name}: gap={metrics['gap']:.4f}, SCI={metrics['SCI']:.3f}")

    return pd.DataFrame(rows)


# -----------------------------
# Plotting
# -----------------------------

def save_plots(df: pd.DataFrame, out_dir: str) -> None:
    gdf = df[df["signal"] == "GARCH(1,1)"].copy()
    if gdf.empty:
        print("No GARCH results to plot.")
        return

    # 1. Gap vs persistence
    plt.figure(figsize=(8, 5))
    plt.scatter(gdf["persistence_alpha_plus_beta"], gdf["gap"])
    plt.xlabel("GARCH persistence alpha + beta")
    plt.ylabel("Envelope-coherence gap: c_obs - mean(c_surr)")
    plt.title("SCI gap vs GARCH volatility persistence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gap_vs_persistence.png"), dpi=200)
    plt.close()

    # 2. SCI vs persistence
    plt.figure(figsize=(8, 5))
    plt.scatter(gdf["persistence_alpha_plus_beta"], gdf["SCI"])
    plt.axhline(0.75, linestyle="--", linewidth=1, label="CORE threshold 0.75")
    plt.axhline(0.65, linestyle="--", linewidth=1, label="CORE_MID threshold 0.65")
    plt.xlabel("GARCH persistence alpha + beta")
    plt.ylabel("SCI")
    plt.title("SCI vs GARCH volatility persistence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sci_vs_persistence.png"), dpi=200)
    plt.close()

    # 3. Gap heatmap
    pivot_gap = gdf.pivot_table(
        index="alpha",
        columns="beta",
        values="gap",
        aggfunc="mean"
    ).sort_index(ascending=True)

    plt.figure(figsize=(9, 5))
    plt.imshow(pivot_gap.values, aspect="auto", origin="lower")
    plt.colorbar(label="Gap")
    plt.xticks(range(len(pivot_gap.columns)), [f"{b:.2f}" for b in pivot_gap.columns], rotation=45)
    plt.yticks(range(len(pivot_gap.index)), [f"{a:.2f}" for a in pivot_gap.index])
    plt.xlabel("beta")
    plt.ylabel("alpha")
    plt.title("Heatmap: envelope-coherence gap by GARCH parameters")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_gap_alpha_beta.png"), dpi=200)
    plt.close()

    # 4. SCI heatmap
    pivot_sci = gdf.pivot_table(
        index="alpha",
        columns="beta",
        values="SCI",
        aggfunc="mean"
    ).sort_index(ascending=True)

    plt.figure(figsize=(9, 5))
    plt.imshow(pivot_sci.values, aspect="auto", origin="lower", vmin=0, vmax=1)
    plt.colorbar(label="SCI")
    plt.xticks(range(len(pivot_sci.columns)), [f"{b:.2f}" for b in pivot_sci.columns], rotation=45)
    plt.yticks(range(len(pivot_sci.index)), [f"{a:.2f}" for a in pivot_sci.index])
    plt.xlabel("beta")
    plt.ylabel("alpha")
    plt.title("Heatmap: SCI by GARCH parameters")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_sci_alpha_beta.png"), dpi=200)
    plt.close()


def print_summary(df: pd.DataFrame) -> None:
    gdf = df[df["signal"] == "GARCH(1,1)"].dropna(subset=["persistence_alpha_plus_beta"])

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if len(gdf) >= 3:
        r_gap, p_gap = pearsonr(gdf["persistence_alpha_plus_beta"], gdf["gap"])
        r_sci, p_sci = pearsonr(gdf["persistence_alpha_plus_beta"], gdf["SCI"])
        r_amp, p_amp = pearsonr(gdf["sq_acf"], gdf["gap"])

        print(f"Correlation: persistence alpha+beta vs gap: r={r_gap:.3f}, p={p_gap:.4g}")
        print(f"Correlation: persistence alpha+beta vs SCI: r={r_sci:.3f}, p={p_sci:.4g}")
        print(f"Correlation: squared-amplitude ACF vs gap: r={r_amp:.3f}, p={p_amp:.4g}")

        print("\nTop 10 GARCH settings by gap:")
        cols = [
            "alpha", "beta", "persistence_alpha_plus_beta",
            "gap", "SCI", "c_obs", "c_surr_mean", "sq_acf"
        ]
        print(
            gdf.sort_values("gap", ascending=False)[cols]
            .head(10)
            .to_string(index=False)
        )

        print("\nBottom 10 GARCH settings by gap:")
        print(
            gdf.sort_values("gap", ascending=True)[cols]
            .head(10)
            .to_string(index=False)
        )

    print("\nBaseline controls:")
    base = df[df["signal"] != "GARCH(1,1)"]
    if not base.empty:
        print(base[["signal", "gap", "SCI", "c_obs", "c_surr_mean", "sq_acf"]].to_string(index=False))

    print("\nInterpretation:")
    print(
        "If gap and SCI rise with alpha+beta, that supports the revised derivation target: "
        "persistent nonlinear amplitude dynamics create envelope coherence that phase-randomized "
        "surrogates do not preserve. If the relationship is weak, the proof target needs more "
        "specific conditions than alpha+beta alone, such as alpha level, shock heaviness, window size, "
        "or volatility half-life."
    )


# -----------------------------
# Main
# -----------------------------

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    params = SCIParams()
    out_dir = "sci_garch_gap_results"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("SCI GARCH GAP MAPPER")
    print("=" * 80)
    print(params)

    garch_df = run_garch_sweep(params)
    baseline_df = run_baselines(params)

    df = pd.concat([garch_df, baseline_df], ignore_index=True)

    csv_path = os.path.join(out_dir, "garch_gap_results.csv")
    df.to_csv(csv_path, index=False)

    save_plots(df, out_dir)
    print_summary(df)

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {os.path.join(out_dir, 'gap_vs_persistence.png')}")
    print(f"  {os.path.join(out_dir, 'sci_vs_persistence.png')}")
    print(f"  {os.path.join(out_dir, 'heatmap_gap_alpha_beta.png')}")
    print(f"  {os.path.join(out_dir, 'heatmap_sci_alpha_beta.png')}")


if __name__ == "__main__":
    main()
