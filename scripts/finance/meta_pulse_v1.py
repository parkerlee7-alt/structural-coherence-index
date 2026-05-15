"""
Meta-Pulse v1 — Cross-envelope coherence among CORE-classified equity time series.

Hypothesis: CORE-classified stocks share a latent amplitude driver. Their Hilbert
envelope envelopes co-move more than phase-randomized surrogates predict.

Algorithm (per window):
  1. Select the top-N CORE stocks by w500_gap (strongest amplitude organization)
  2. For each stock compute the smoothed Hilbert amplitude envelope of daily returns
  3. Observe: mean pairwise Pearson correlation across all N(N-1)/2 envelope pairs
  4. Surrogate: phase-randomize each return series independently (preserving each
     series' own power spectrum); re-compute envelopes; re-compute mean pairwise corr
     Repeat S times → null distribution
  5. Meta-Pulse gap  = c_obs − c_surr_mean
     Meta-Pulse z    = gap / c_surr_std
     Meta-Pulse SCI  = 1 / (1 + exp(−0.9 × z))   (same logistic as SCI)

Two experiments:
  A. Full-period: one Meta-Pulse computation on the top-N CORE stocks over all
     available history (quick validation)
  B. Rolling: slide a 500-day window across the data; at each step pick top-N CORE
     stocks by SCI computed in that window; plot Meta-Pulse index over time

Usage:
    python3 scripts/finance/meta_pulse_v1.py \
        --prices  pulse_garch_finance/cache_prices \
        --results pulse_garch_finance/results_garch_finance/financial_garch_sci_results_v2_ranked.csv \
        --out     results/meta_pulse_v1 \
        --top-n   20 \
        --surrogates 200 \
        --rolling
"""

import sys, os, argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.signal as sg
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from sci_score_v3 import sci_score_v3

# ── locked finance parameters ─────────────────────────────────────────────────
W_SMOOTH  = 7        # envelope smoothing (finance: 7-day)
WINDOW    = 500      # rolling SCI window (trading days)
S_DEFAULT = 200      # surrogates for Meta-Pulse null
K         = 0.9
SEED      = 42
MIN_STOCKS = 5       # minimum CORE stocks required to compute Meta-Pulse


# ── helpers ──────────────────────────────────────────────────────────────────

def phase_randomize(x, rng):
    """Phase-randomize x preserving power spectrum. x must be 1-D."""
    n    = len(x)
    ft   = np.fft.rfft(x)
    mag  = np.abs(ft)
    # random phases, conjugate-symmetric
    phases = rng.uniform(0, 2 * np.pi, len(mag))
    phases[0]  = 0.0           # DC
    if n % 2 == 0:
        phases[-1] = 0.0       # Nyquist
    ft_rand = mag * np.exp(1j * phases)
    return np.fft.irfft(ft_rand, n=n)


def hilbert_envelope(x, w=W_SMOOTH):
    """Smoothed Hilbert amplitude envelope."""
    env = np.abs(sg.hilbert(x))
    if w > 1:
        env = np.convolve(env, np.ones(w) / w, mode="same")
    return env


def mean_pairwise_corr(envelopes):
    """Mean Pearson r across all unique pairs of envelopes."""
    pairs = list(combinations(range(len(envelopes)), 2))
    if not pairs:
        return np.nan
    corrs = [pearsonr(envelopes[i], envelopes[j])[0] for i, j in pairs]
    return float(np.nanmean(corrs))


def meta_pulse(returns_matrix, S=S_DEFAULT, seed=SEED, w=W_SMOOTH):
    """
    Compute Meta-Pulse for a matrix of returns.

    Parameters
    ----------
    returns_matrix : np.ndarray, shape (T, N)
        T time points, N stocks
    S : int
        Number of phase-randomized surrogates
    seed : int
        RNG seed
    w : int
        Envelope smoothing window

    Returns
    -------
    dict with keys: c_obs, c_surr_mean, c_surr_std, gap, z, SCI, n_stocks, n_pairs
    """
    T, N = returns_matrix.shape
    rng  = np.random.default_rng(seed)

    # observed envelopes + mean pairwise correlation
    envelopes = [hilbert_envelope(returns_matrix[:, i], w=w) for i in range(N)]
    c_obs     = mean_pairwise_corr(envelopes)

    # surrogate null
    c_surr = np.zeros(S)
    for s in range(S):
        surr_env = [
            hilbert_envelope(phase_randomize(returns_matrix[:, i], rng), w=w)
            for i in range(N)
        ]
        c_surr[s] = mean_pairwise_corr(surr_env)

    gap  = c_obs - c_surr.mean()
    z    = np.clip(gap / (c_surr.std() + 1e-12), -6, 6)
    sci  = 1.0 / (1.0 + np.exp(-K * z))
    n_pairs = N * (N - 1) // 2

    return {
        "c_obs":       c_obs,
        "c_surr_mean": c_surr.mean(),
        "c_surr_std":  c_surr.std(),
        "gap":         gap,
        "z":           z,
        "meta_pulse":  sci,
        "n_stocks":    N,
        "n_pairs":     n_pairs,
    }


def load_prices(prices_dir, tickers):
    """Load daily close prices for a list of tickers. Returns aligned DataFrame."""
    frames = {}
    for t in tickers:
        p = Path(prices_dir) / f"{t}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        col = "Close" if "Close" in df.columns else df.columns[0]
        frames[t] = df[col].rename(t)
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).sort_index().dropna(how="all")


def rolling_sci_gap(returns, w_smooth=W_SMOOTH, window=WINDOW, S=40, seed=SEED):
    """
    Compute rolling SCI gap for a single return series.
    Returns Series of gap values aligned to the right edge of each window.
    """
    n   = len(returns)
    idx = returns.index
    gaps = {}
    for end in range(window, n + 1):
        seg  = returns.iloc[end - window:end].values.astype(float)
        seg  = seg[np.isfinite(seg)]
        if len(seg) < window // 2:
            continue
        r    = sci_score_v3(seg, w=w_smooth, L=10, S=S, k=K, seed=seed)
        gaps[idx[end - 1]] = r["gap"]
    return pd.Series(gaps)


# ── Experiment A: full-period ────────────────────────────────────────────────

def experiment_a(results_df, prices_dir, top_n, S, out_dir):
    print("\n=== Experiment A: Full-period Meta-Pulse ===")

    # select top-N CORE by w500_gap
    core = results_df[results_df["amplitude_bucket_v2"].str.contains("CORE", na=False)].copy()
    if len(core) < MIN_STOCKS:
        # fall back to top-N by gap
        core = results_df.nlargest(top_n * 2, "w500_gap")
    core = core.nlargest(top_n, "w500_gap")
    tickers = core["ticker"].tolist()
    print(f"  Tickers selected ({len(tickers)}): {tickers}")

    prices = load_prices(prices_dir, tickers)
    if prices.empty:
        print("  ERROR: no price data found.")
        return
    prices = prices.dropna(thresh=len(prices.columns) // 2)
    returns = prices.pct_change().dropna()

    # align — keep only dates where all stocks have data
    returns = returns[tickers].dropna()
    print(f"  Date range: {returns.index[0].date()} → {returns.index[-1].date()}  "
          f"({len(returns)} days)")

    result = meta_pulse(returns.values, S=S, w=W_SMOOTH)
    result["tickers"] = tickers

    print(f"\n  Meta-Pulse results:")
    print(f"    c_obs        = {result['c_obs']:.4f}")
    print(f"    c_surr_mean  = {result['c_surr_mean']:.4f}")
    print(f"    gap          = {result['gap']:.4f}")
    print(f"    z            = {result['z']:.4f}")
    print(f"    Meta-Pulse   = {result['meta_pulse']:.4f}")
    print(f"    n_stocks     = {result['n_stocks']}  |  n_pairs = {result['n_pairs']}")

    # ── surrogate distribution plot ───────────────────────────────────────────
    rng  = np.random.default_rng(SEED)
    c_surr_dist = np.zeros(S)
    envelopes = [hilbert_envelope(returns.values[:, i]) for i in range(len(tickers))]
    c_obs_val = mean_pairwise_corr(envelopes)
    for s in range(S):
        surr = [hilbert_envelope(phase_randomize(returns.values[:, i], rng))
                for i in range(len(tickers))]
        c_surr_dist[s] = mean_pairwise_corr(surr)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # surrogate distribution
    ax = axes[0]
    ax.hist(c_surr_dist, bins=30, color="steelblue", alpha=0.7, label="Surrogates")
    ax.axvline(c_obs_val, color="red", linewidth=2, label=f"Observed = {c_obs_val:.4f}")
    ax.set_xlabel("Mean pairwise envelope correlation")
    ax.set_ylabel("Count")
    ax.set_title(f"Meta-Pulse Null Distribution\n"
                 f"gap={result['gap']:.4f}  z={result['z']:.2f}  "
                 f"Meta-Pulse={result['meta_pulse']:.4f}")
    ax.legend()

    # envelope overlay for top-5 stocks
    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, min(5, len(tickers))))
    for i, (t, col) in enumerate(zip(tickers[:5], colors)):
        env = hilbert_envelope(returns.values[:, i])
        env = (env - env.mean()) / (env.std() + 1e-12)  # z-score for overlay
        ax.plot(returns.index, env, alpha=0.6, linewidth=0.8, color=col, label=t)
    ax.set_title(f"Hilbert Envelopes — Top 5 CORE Stocks\n(z-scored for overlay)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Envelope (z-scored)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(out_dir / "meta_pulse_full_period.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved → meta_pulse_full_period.png")

    # save summary
    summary = pd.DataFrame([{
        "experiment":  "full_period",
        "n_stocks":    result["n_stocks"],
        "n_pairs":     result["n_pairs"],
        "c_obs":       result["c_obs"],
        "c_surr_mean": result["c_surr_mean"],
        "c_surr_std":  result["c_surr_std"],
        "gap":         result["gap"],
        "z":           result["z"],
        "meta_pulse":  result["meta_pulse"],
        "tickers":     ", ".join(tickers),
    }])
    summary.to_csv(out_dir / "meta_pulse_full_period.csv", index=False)
    return result


# ── Experiment B: rolling ────────────────────────────────────────────────────

def experiment_b(results_df, prices_dir, top_n, S, out_dir, step=21):
    """Rolling Meta-Pulse — recompute every `step` trading days."""
    print(f"\n=== Experiment B: Rolling Meta-Pulse (window={WINDOW}, step={step}) ===")

    # load ALL CORE tickers
    core_tickers = results_df[
        results_df["amplitude_bucket_v2"].str.contains("CORE", na=False)
    ]["ticker"].tolist()

    if len(core_tickers) < MIN_STOCKS:
        core_tickers = results_df.nlargest(top_n * 3, "w500_gap")["ticker"].tolist()

    print(f"  Loading prices for {len(core_tickers)} CORE tickers...")
    prices = load_prices(prices_dir, core_tickers)
    if prices.empty:
        print("  ERROR: no price data found.")
        return

    prices = prices.dropna(thresh=max(MIN_STOCKS, len(prices.columns) // 4))
    returns_all = prices.pct_change().dropna(how="all")

    dates = returns_all.index
    n     = len(dates)
    if n < WINDOW + step:
        print(f"  Not enough data ({n} days). Need >{WINDOW}.")
        return

    records = []
    eval_dates = range(WINDOW, n, step)
    total = len(list(eval_dates))
    print(f"  Computing Meta-Pulse at {total} windows...")

    for k, end in enumerate(range(WINDOW, n, step)):
        window_ret = returns_all.iloc[end - WINDOW:end]
        # drop stocks with >10% missing in window
        valid = window_ret.columns[window_ret.isna().mean() < 0.1]
        window_ret = window_ret[valid].fillna(0)

        if len(valid) < MIN_STOCKS:
            continue

        # pick top-N by in-window SCI gap
        gaps = {}
        for t in valid:
            seg = window_ret[t].values.astype(float)
            try:
                r = sci_score_v3(seg, w=W_SMOOTH, L=10, S=20, k=K, seed=SEED)
                gaps[t] = r["gap"]
            except Exception:
                gaps[t] = -999

        top_tickers = sorted(gaps, key=gaps.get, reverse=True)[:top_n]
        sub_ret     = window_ret[top_tickers].values.astype(float)

        if sub_ret.shape[1] < MIN_STOCKS:
            continue

        result = meta_pulse(sub_ret, S=S, w=W_SMOOTH)
        records.append({
            "date":        dates[end - 1],
            "n_stocks":    result["n_stocks"],
            "c_obs":       result["c_obs"],
            "c_surr_mean": result["c_surr_mean"],
            "gap":         result["gap"],
            "z":           result["z"],
            "meta_pulse":  result["meta_pulse"],
            "top_tickers": ", ".join(top_tickers[:5]),
        })

        if (k + 1) % 10 == 0 or k == total - 1:
            print(f"  [{k+1}/{total}]  {dates[end-1].date()}  "
                  f"MP={result['meta_pulse']:.3f}  gap={result['gap']:.4f}")

    if not records:
        print("  No windows computed.")
        return

    roll_df = pd.DataFrame(records).set_index("date")
    roll_df.to_csv(out_dir / "meta_pulse_rolling.csv")
    print(f"\n  Saved → meta_pulse_rolling.csv ({len(roll_df)} windows)")

    # ── rolling plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax = axes[0]
    ax.plot(roll_df.index, roll_df["meta_pulse"], color="darkblue", linewidth=1.2)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.fill_between(roll_df.index, 0.5, roll_df["meta_pulse"],
                    where=roll_df["meta_pulse"] > 0.5,
                    color="steelblue", alpha=0.25)
    ax.set_ylabel("Meta-Pulse SCI")
    ax.set_title("Rolling Meta-Pulse — Cross-Envelope Coherence Among CORE Stocks")
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.plot(roll_df.index, roll_df["gap"], color="tomato", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.fill_between(roll_df.index, 0, roll_df["gap"],
                    where=roll_df["gap"] > 0,
                    color="tomato", alpha=0.25)
    ax.set_ylabel("Meta-Pulse gap")
    ax.set_title("Meta-Pulse Gap (c_obs − c_surr_mean)")

    ax = axes[2]
    ax.plot(roll_df.index, roll_df["z"], color="purple", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(2, color="green", linestyle="--", linewidth=0.7, alpha=0.7,
               label="z=2")
    ax.set_ylabel("z-score")
    ax.set_title("Meta-Pulse z-score")
    ax.legend(fontsize=8)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    fig.savefig(out_dir / "meta_pulse_rolling.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → meta_pulse_rolling.png")

    # summary stats
    print(f"\n  Rolling Meta-Pulse summary:")
    print(f"    Mean MP:    {roll_df['meta_pulse'].mean():.4f}")
    print(f"    Max  MP:    {roll_df['meta_pulse'].max():.4f}  ({roll_df['meta_pulse'].idxmax().date()})")
    print(f"    % MP > 0.5: {(roll_df['meta_pulse'] > 0.5).mean()*100:.1f}%")
    print(f"    Mean gap:   {roll_df['gap'].mean():.4f}")

    return roll_df


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Meta-Pulse: cross-envelope coherence")
    parser.add_argument("--prices",     default="pulse_garch_finance/cache_prices")
    parser.add_argument("--results",    default="pulse_garch_finance/results_garch_finance/financial_garch_sci_results_v2_ranked.csv")
    parser.add_argument("--out",        default="results/meta_pulse_v1")
    parser.add_argument("--top-n",      type=int, default=20)
    parser.add_argument("--surrogates", type=int, default=200)
    parser.add_argument("--rolling",    action="store_true")
    parser.add_argument("--step",       type=int, default=21,
                        help="Rolling step in trading days (default 21 = ~1 month)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(args.results)
    print(f"Loaded {len(results_df)} tickers from SCI results.")

    # Experiment A — full period
    experiment_a(results_df, args.prices, args.top_n, args.surrogates, out_dir)

    # Experiment B — rolling
    if args.rolling:
        experiment_b(results_df, args.prices, args.top_n,
                     max(40, args.surrogates // 5),  # fewer surrogates for speed
                     out_dir, step=args.step)

    print(f"\nAll done. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
